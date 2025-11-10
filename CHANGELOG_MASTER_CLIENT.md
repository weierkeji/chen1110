# Master Client 功能添加 - 修改日志

## 修改概述

本次更新为所有数据收集器（Data Collectors）添加了 Master Client 支持，使其能够将收集到的诊断数据上报到中控节点（Master）。

## 修改的文件

### 1. `agent/data_collector/log_collector.py`

**修改内容：**
- ✅ 在 `__init__` 中添加 `self._client = None` 成员变量
- ✅ 添加 `set_client(client)` 方法用于设置 master client
- ✅ 修改 `store_data()` 方法：
  - 添加 `if not data: return` 检查（避免上报空数据）
  - 添加实际的上报逻辑：`self._client.report_diagnosis_agent_metrics(agent_log_metric)`
  - 添加异常处理（try-except）
  - 根据是否有 client 记录不同的日志消息

**修改前：**
```python
def store_data(self, data: object):
    # ...
    agent_log_metric = WorkerTrainingMetric(...)
    # In a real implementation, this would report to master
    logger.info(f"Collected log content: {len(data)} characters")
```

**修改后：**
```python
def store_data(self, data: object):
    # ...
    agent_log_metric = WorkerTrainingMetric(...)
    
    # Report to master if client is available
    if self._client:
        try:
            self._client.report_diagnosis_agent_metrics(agent_log_metric)
            logger.info(f"Reported log content: {len(data)} characters")
        except Exception as e:
            logger.error(f"Failed to report log data to master: {e}")
    else:
        logger.warning("Master client not set, log data not reported")
```

---

### 2. `agent/data_collector/metric_collector.py`

**修改内容：**
- ✅ 修改 `__init__` 中的注释：`# Master client for reporting data`
- ✅ 添加 `set_client(client)` 方法
- ✅ 修改 `store_data()` 方法（与 log_collector 相同的模式）：
  - 添加空数据检查
  - 实现实际上报
  - 异常处理
  - 条件日志记录

**数据类型：** `DiagnosisDataType.XPU_TIMER_METRIC`

---

### 3. `agent/data_collector/stack_collector.py`

**修改内容：**
- ✅ 在 `__init__` 中添加 `self._client = None` 成员变量
- ✅ 添加 `set_client(client)` 方法
- ✅ 修改 `store_data()` 方法（与其他 collectors 相同的模式）

**数据类型：** `DiagnosisDataType.STACK_TRACE`

---

### 4. `controller/diagnosis.py`

**修改内容：**

#### a) 在 `__init__` 中添加 master client 支持
```python
# Master client for reporting data
self._client = None
```

#### b) 添加 `set_client(client)` 方法
```python
def set_client(self, client):
    """
    Set master client for reporting data.
    
    Args:
        client: Master client instance.
    """
    self._client = client
    logger.info("Master client set for diagnosis agent")
    
    # Set client for all registered collectors
    for collector in self._periodical_collectors.keys():
        if hasattr(collector, "set_client"):
            collector.set_client(client)
            logger.debug(f"Set client for {collector.__class__.__name__}")
```

**功能：**
- 设置 DiagnosisAgent 自己的 client
- 自动为所有已注册的 collectors 设置 client
- 使用 `hasattr` 检查，避免对不支持的 collector 报错

#### c) 修改 `register_periodical_data_collector()` 方法
```python
def register_periodical_data_collector(self, collector: DataCollector, time_interval: int):
    # ...
    self._periodical_collectors[collector] = time_interval
    
    # Set client if already available
    if self._client and hasattr(collector, "set_client"):
        collector.set_client(self._client)
    
    logger.info(...)
```

**功能：**
- 在注册 collector 时，如果 agent 已经有 client，自动设置给 collector
- 这样即使在 `set_client()` 之后注册的 collector 也能正确获得 client

---

## 新增文件

### 1. `MASTER_CLIENT_USAGE.md`

详细的使用文档，包含：
- 架构说明
- 两种使用方式（通过 DiagnosisAgent 和手动设置）
- 数据上报流程详解
- 数据格式说明
- Master Client 接口要求
- 完整示例代码
- 错误处理说明
- 与 DLRover 的兼容性说明

### 2. `CHANGELOG_MASTER_CLIENT.md`

本文档，详细记录所有修改内容。

---

## 核心设计模式

### 1. 依赖注入模式
```python
# 不在 collector 内部创建 client，而是从外部注入
collector.set_client(master_client)
```

**优点：**
- 解耦：collector 不依赖具体的 client 实现
- 灵活：可以注入不同的 client（mock client for testing）
- 可测试：方便单元测试

### 2. 防御性编程
```python
# 检查 client 是否存在
if self._client:
    try:
        self._client.report_diagnosis_agent_metrics(data)
    except Exception as e:
        logger.error(f"Failed to report: {e}")
else:
    logger.warning("Client not set")
```

**优点：**
- 健壮性：即使没有设置 client，也不会崩溃
- 可观测性：通过日志可以知道是否正确上报
- 容错性：上报失败不影响训练继续进行

### 3. 统一接口
```python
# 所有 collectors 都有相同的接口
def set_client(self, client): ...
def store_data(self, data): ...
```

**优点：**
- 一致性：所有 collectors 使用方式相同
- 可扩展性：新增 collector 遵循相同模式
- 可维护性：代码结构清晰统一

---

## 数据流图

```
┌─────────────────────┐
│  Training Process   │
│                     │
│  写日志、产生指标    │
└──────────┬──────────┘
           │
           │ 周期性触发
           ▼
┌─────────────────────┐
│  Data Collectors    │
│                     │
│  - collect_data()   │ ──┐
│  - store_data()     │   │ 收集本地数据
└──────────┬──────────┘   │
           │               │
           │               │
           ▼               ▼
┌─────────────────────────────┐
│  WorkerTrainingMetric       │
│  {                          │
│    data_type: TRAINING_LOG  │
│    data_content: "...",     │
│    node_id: 0,              │
│    node_type: "worker",     │
│    node_rank: 0,            │
│    timestamp: 1234567890    │
│  }                          │
└──────────┬──────────────────┘
           │
           │ _client.report_diagnosis_agent_metrics()
           ▼
┌─────────────────────┐
│   Master Client     │
│                     │
│  序列化、网络传输    │
└──────────┬──────────┘
           │
           │ gRPC / HTTP
           ▼
┌─────────────────────┐
│   Master Node       │
│                     │
│  - 接收数据         │
│  - 存储分析         │
│  - 故障诊断         │
└─────────────────────┘
```

---

## 使用示例

### 基础用法

```python
from arobust.controller.diagnosis import DiagnosisAgent
from your_master_client import MasterClient

# 1. 创建 master client
client = MasterClient(master_addr="localhost:50051")

# 2. 创建并配置 agent
agent = DiagnosisAgent.singleton_instance(
    training_log_file="/path/to/train.log"
)

# 3. 设置 client（会自动传递给所有 collectors）
agent.set_client(client)

# 4. 启动（开始自动收集和上报）
agent.start()

# 5. 训练...

# 6. 停止
agent.stop()
```

### 单独使用 Collector

```python
from arobust.agent.data_collector import LogCollector

collector = LogCollector("/path/to/train.log")
collector.set_client(master_client)

data = collector.collect_data()
collector.store_data(data)  # 自动上报
```

---

## 测试建议

### 1. 单元测试

测试 collector 的 store_data 方法：

```python
def test_log_collector_with_client():
    # 创建 mock client
    mock_client = Mock()
    
    # 创建 collector 并设置 client
    collector = LogCollector("/path/to/log")
    collector.set_client(mock_client)
    
    # 调用 store_data
    collector.store_data("test log content")
    
    # 验证 client 被调用
    mock_client.report_diagnosis_agent_metrics.assert_called_once()
    
    # 验证参数类型
    args = mock_client.report_diagnosis_agent_metrics.call_args[0]
    assert isinstance(args[0], WorkerTrainingMetric)
    assert args[0].data_type == DiagnosisDataType.TRAINING_LOG
```

### 2. 集成测试

测试完整的数据流：

```python
def test_diagnosis_agent_integration():
    # 创建真实的 master client
    client = MasterClient(master_addr="localhost:50051")
    
    # 创建 agent
    agent = DiagnosisAgent.singleton_instance()
    agent.set_client(client)
    agent.start()
    
    # 等待一段时间，让 collectors 运行
    time.sleep(5)
    
    # 检查 master 是否收到数据
    # ...
    
    agent.stop()
```

---

## 向后兼容性

✅ **完全向后兼容**

- 如果不调用 `set_client()`，collectors 仍然可以正常工作
- 只是不会上报数据到 master，而是记录警告日志
- 不会抛出任何异常
- 不影响现有的训练代码

---

## 已知限制

1. **ResourceCollector 未修改**
   - 因为 ResourceCollector 直接调用 ResourceMonitor，没有自己的 store_data 逻辑
   - ResourceMonitor 有自己的上报机制

2. **需要实现 MasterClient**
   - 本次只添加了接口调用
   - 实际的 MasterClient 实现需要根据你的通信协议（gRPC/HTTP）单独实现
   - 可以参考 dlrover 的实现

---

## 下一步工作

1. **实现 MasterClient**
   - 基于 gRPC 或 HTTP
   - 实现 `report_diagnosis_agent_metrics()` 方法
   - 添加重试和超时机制

2. **添加更多测试**
   - 单元测试
   - 集成测试
   - 性能测试

3. **优化数据传输**
   - 批量上报（减少网络请求）
   - 数据压缩
   - 异步上报

4. **添加监控指标**
   - 上报成功率
   - 上报延迟
   - 数据大小统计

---

## 参考资料

- **DLRover 源码**
  - `dlrover/python/elastic_agent/master_client.py` - Master Client 实现
  - `dlrover/python/diagnosis/datacollector/` - Data Collectors 实现
  - `dlrover/python/diagnosis/common/diagnosis_data.py` - 数据结构定义

- **设计文档**
  - `MASTER_CLIENT_USAGE.md` - 使用指南
  - `.plan.md` - 项目整体规划

---

## 修改统计

- **修改文件数**: 4
- **新增文件数**: 2
- **新增代码行数**: ~150 行
- **修改代码行数**: ~80 行
- **新增方法**: 4 个 `set_client()` 方法
- **修改方法**: 3 个 `store_data()` 方法

---

## 审核检查清单

- [x] 所有 collectors 都添加了 `set_client()` 方法
- [x] 所有 collectors 的 `store_data()` 都实现了实际上报
- [x] DiagnosisAgent 添加了 `set_client()` 方法
- [x] DiagnosisAgent 会自动为 collectors 设置 client
- [x] 添加了异常处理，避免上报失败影响训练
- [x] 添加了完善的日志记录
- [x] 保持向后兼容性（不设置 client 也能运行）
- [x] 编写了详细的使用文档
- [x] 遵循 DLRover 的设计模式

---

**修改完成时间**: 2025-11-10
**修改人员**: AI Assistant
**审核状态**: ✅ 已完成

