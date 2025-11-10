# Master Client 使用指南

本文档说明如何在 arobust 系统中使用 Master Client 进行数据上报。

## 概述

在 arobust 中，数据收集器（Data Collectors）通过 Master Client 向中控节点（Master）上报诊断数据。此功能已在以下收集器中实现：

- `LogCollector` - 日志收集器
- `MetricCollector` - XPUTimer 指标收集器  
- `StackCollector` - 堆栈追踪收集器

## 架构说明

```
┌─────────────────────────┐
│   DiagnosisAgent        │  设置 Master Client
│   (顶层编排器)          │ ─────────────┐
└─────────────────────────┘               │
            │                             │
            │ 注册 collectors              │
            ▼                             ▼
┌──────────────────────────────────────────────┐
│         Data Collectors                      │
│  - LogCollector                              │
│  - MetricCollector          使用 client      │
│  - StackCollector           上报数据         │
└──────────────────────────────────────────────┘
            │
            │ report_diagnosis_agent_metrics()
            ▼
┌─────────────────────────┐
│    Master Client        │  发送数据到 Master
└─────────────────────────┘
```

## 使用方式

### 方式一：通过 DiagnosisAgent（推荐）

DiagnosisAgent 会自动为所有注册的 collector 设置 master client：

```python
from arobust.controller.diagnosis import DiagnosisAgent
from your_master_client import MasterClient  # 你的 Master Client 实现

# 创建 Master Client
master_client = MasterClient(master_addr="localhost:50051")

# 创建并配置 DiagnosisAgent
agent = DiagnosisAgent.singleton_instance(
    training_log_file="/path/to/training.log",
    node_rank=0
)

# 设置 Master Client（会自动传递给所有 collectors）
agent.set_client(master_client)

# 启动 agent（此时所有 collectors 都已配置好 client）
agent.start()
```

### 方式二：手动设置单个 Collector

如果只使用单个 collector，可以手动设置：

```python
from arobust.agent.data_collector import LogCollector
from your_master_client import MasterClient

# 创建 Master Client
master_client = MasterClient(master_addr="localhost:50051")

# 创建 Collector
log_collector = LogCollector(log_file_path="/path/to/training.log")

# 设置 Master Client
log_collector.set_client(master_client)

# 使用 collector
if log_collector.is_enabled():
    data = log_collector.collect_data()
    if data:
        log_collector.store_data(data)  # 会自动上报到 Master
```

## 数据上报流程

### 1. 日志数据上报（LogCollector）

```python
# LogCollector.store_data() 内部实现：
def store_data(self, data: str):
    # 创建 WorkerTrainingMetric 对象
    agent_log_metric = WorkerTrainingMetric(
        data_type=DiagnosisDataType.TRAINING_LOG,
        data_content=data,  # 日志内容字符串
        node_id=get_node_id(),
        node_type=get_node_type(),
        node_rank=get_node_rank(),
    )
    
    # 通过 Master Client 上报
    if self._client:
        self._client.report_diagnosis_agent_metrics(agent_log_metric)
```

### 2. XPU 指标数据上报（MetricCollector）

```python
# MetricCollector.store_data() 内部实现：
def store_data(self, data: str):
    # 创建 WorkerTrainingMetric 对象
    agent_xpu_metric = WorkerTrainingMetric(
        data_type=DiagnosisDataType.XPU_TIMER_METRIC,
        data_content=data,  # XPU 指标字符串
        node_id=get_node_id(),
        node_type=get_node_type(),
        node_rank=get_node_rank(),
    )
    
    # 通过 Master Client 上报
    if self._client:
        self._client.report_diagnosis_agent_metrics(agent_xpu_metric)
```

### 3. 堆栈追踪数据上报（StackCollector）

```python
# StackCollector.store_data() 内部实现：
def store_data(self, data: dict):
    # 格式化堆栈数据
    stack_str = self._format_stacks(data)
    
    # 创建 WorkerTrainingMetric 对象
    agent_stack_metric = WorkerTrainingMetric(
        data_type=DiagnosisDataType.STACK_TRACE,
        data_content=stack_str,  # 堆栈追踪字符串
        node_id=get_node_id(),
        node_type=get_node_type(),
        node_rank=get_node_rank(),
    )
    
    # 通过 Master Client 上报
    if self._client:
        self._client.report_diagnosis_agent_metrics(agent_stack_metric)
```

## 数据格式

所有诊断数据都使用 `WorkerTrainingMetric` 类进行封装，包含以下字段：

```python
@dataclass
class WorkerTrainingMetric:
    data_type: DiagnosisDataType      # 数据类型（TRAINING_LOG, XPU_TIMER_METRIC, STACK_TRACE）
    data_content: str                  # 数据内容（字符串格式）
    node_id: int                       # 节点ID
    node_type: str                     # 节点类型
    node_rank: int                     # 节点Rank
    timestamp: int                     # 时间戳（自动生成）
```

## Master Client 接口要求

你的 Master Client 实现需要提供以下接口：

```python
class MasterClient:
    def report_diagnosis_agent_metrics(self, data: WorkerTrainingMetric):
        """
        上报诊断数据到 Master。
        
        Args:
            data: WorkerTrainingMetric 对象，包含诊断数据
        """
        # 实现数据序列化和网络传输
        # 参考 dlrover 的实现：
        # message = comm.DiagnosisReportData(
        #     data.__class__.__name__,
        #     data.to_json(),
        #     data.node_rank,
        # )
        # self._report(message)
        pass
```

## 完整示例

```python
import logging
import time
from arobust.controller.diagnosis import DiagnosisAgent
from your_project.master_client import MasterClient

# 配置日志
logging.basicConfig(level=logging.INFO)

# 1. 创建 Master Client
master_client = MasterClient(
    master_addr="localhost:50051",
    node_id=0,
    node_type="worker"
)

# 2. 创建 DiagnosisAgent
agent = DiagnosisAgent.singleton_instance(
    training_log_file="/var/log/training.log",
    node_rank=0,
    local_world_size=4
)

# 3. 设置 Master Client
agent.set_client(master_client)

# 4. 启动 Agent
agent.start()

# 5. 训练过程中，collectors 会自动收集和上报数据
try:
    # 你的训练代码
    for epoch in range(100):
        train_one_epoch()
        time.sleep(1)
        
except Exception as e:
    # 6. 发生错误时，可以进行故障诊断
    action = agent.diagnose_training_failure(
        failures={"error": str(e)},
        restart_count=0
    )
    print(f"Diagnosis action: {action.action_type.name}")
    
finally:
    # 7. 停止 Agent
    agent.stop()
```

## 错误处理

collectors 的 `store_data()` 方法包含完善的错误处理：

- **Master Client 未设置**：记录警告日志，不会抛出异常
- **数据上报失败**：捕获异常并记录错误日志，不会中断训练
- **数据格式错误**：记录警告日志并返回

示例日志输出：

```
INFO - Reported log content: 1024 characters
WARNING - Master client not set, log data not reported
ERROR - Failed to report log data to master: Connection refused
```

## 与 DLRover 的兼容性

本实现参考了 DLRover 的设计，与 DLRover 的 Master Client 接口兼容。如果你使用 DLRover 的 Master Client，可以直接使用：

```python
from dlrover.python.elastic_agent.master_client import MasterClient
from arobust.controller.diagnosis import DiagnosisAgent

# 使用 DLRover 的 Master Client
master_client = MasterClient.singleton_instance(
    master_addr="localhost:50051"
)

# 设置到 DiagnosisAgent
agent = DiagnosisAgent.singleton_instance()
agent.set_client(master_client)
```

## 总结

1. **推荐使用 DiagnosisAgent 进行统一管理**，它会自动处理所有 collectors 的 client 设置
2. **Master Client 是可选的**，如果不设置，collectors 只会记录日志而不会上报数据
3. **所有上报操作都有错误处理**，不会影响训练进程的正常运行
4. **数据格式统一使用 WorkerTrainingMetric**，便于 Master 端统一处理

## 参考资料

- DLRover Master Client: `dlrover/python/elastic_agent/master_client.py`
- DLRover 诊断数据: `dlrover/python/diagnosis/common/diagnosis_data.py`
- Arobust 数据收集层: `chen1110/agent/data_collector/`

