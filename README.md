# chen1110 - RL Training Fault Tolerance System

一个完整的强化学习训练容错系统，参考 DLRover 架构设计，提供三层架构：数据收集层、监控层和诊断管理层。

## 特性

- 🔍 **多层数据收集**：资源监控、性能指标、日志分析、堆栈追踪
- 📊 **实时监控**：独立监控线程，周期性采集和上报训练状态
- 🛠️ **智能诊断**：自动检测故障并提供诊断建议
- 💾 **检查点管理**：支持 RL 特定的检查点策略（Actor、Critic、Rollout）
- 🔄 **容错恢复**：自动故障检测和恢复机制

## 架构设计

```
chen1110/
├── agent/                          # Agent 端组件
│   ├── data_collector/            # 数据收集层（底层）
│   │   ├── data_collector.py      # 抽象基类
│   │   ├── resource_collector.py  # 资源收集器
│   │   ├── metric_collector.py    # XPUTimer 指标收集器
│   │   ├── stack_collector.py     # 堆栈收集器
│   │   └── log_collector.py       # 日志收集器
│   └── monitor/                   # 监控层（中间层）
│       ├── resource.py            # 资源监控器
│       └── training.py            # 训练进度监控器
├── controller/                     # Controller 端组件
│   ├── diagnosis.py               # 诊断代理（顶层编排）
│   └── data_manager.py            # 数据管理器
├── ckpt_manager/                  # 检查点管理
│   ├── latest_checkpoint.py       # 最新检查点管理
│   ├── periodic_checkpoint.py     # 周期性检查点
│   ├── ref_logp_ckpt.py          # Reference LogP 检查点
│   └── rollout_response_checkpoint.py  # Rollout 响应检查点
└── common/                        # 公共组件
    ├── constants.py              # 常量定义
    ├── diagnosis_data.py         # 诊断数据结构
    └── utils.py                  # 工具函数
```

## 安装

### 从源码安装

```bash
cd chen1110
pip install -e .
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 可选：与 DLRover 集成

```bash
pip install -e ".[dlrover]"
```

## 快速开始

### 1. 基础使用

```python
from chen1110 import ResourceMonitor, TrainingMonitor, DiagnosisAgent

# 启动资源监控
resource_monitor = ResourceMonitor.singleton_instance()
resource_monitor.start()

# 启动训练监控
training_monitor = TrainingMonitor.singleton_instance(
    metrics_path="/tmp/metrics.json"
)
training_monitor.start()

# 启动诊断代理
diagnosis_agent = DiagnosisAgent.singleton_instance(
    training_log_file="/tmp/training.log"
)
```

### 2. 数据收集

```python
from chen1110.agent.data_collector import (
    ResourceCollector,
    MetricCollector,
    LogCollector,
)

# 资源收集
resource_collector = ResourceCollector()
if resource_collector.is_enabled():
    data = resource_collector.collect_data()
    resource_collector.store_data(data)

# 性能指标收集（需要 XPUTimer）
metric_collector = MetricCollector()
if metric_collector.is_enabled():
    metrics = metric_collector.collect_data()
    metric_collector.store_data(metrics)
```

### 3. 检查点管理

```python
from chen1110.ckpt_manager import LatestCheckpointManager

# 创建检查点管理器
ckpt_manager = LatestCheckpointManager(
    checkpoint_dir="/path/to/checkpoints",
    max_checkpoints=3
)

# 保存检查点
state_dict = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": global_step,
}
ckpt_manager.save(state_dict, step=global_step)

# 加载检查点
state_dict = ckpt_manager.load()
if state_dict:
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
```

## API 文档

### ResourceMonitor

资源监控器，周期性收集 CPU、内存、GPU 使用情况。

```python
class ResourceMonitor(Singleton):
    def __init__(self, gpu_type: str = "NVIDIA_GPU")
    def start(self) -> None
    def stop(self) -> None
    def report_resource(self) -> None
```

### TrainingMonitor

训练监控器，监控训练进度并上报。

```python
class TrainingMonitor(Singleton):
    def __init__(self, metrics_path: str, device_type: str = "NVIDIA_GPU")
    def start(self) -> None
    def stop(self) -> None
    def report_step(self) -> None
```

### DiagnosisAgent

诊断代理，整合数据收集和诊断功能。

```python
class DiagnosisAgent(Singleton):
    def __init__(self, training_log_file: str = "", errors: str = "")
    def start(self) -> None
    def stop(self) -> None
    def diagnose_training_failure(self) -> DiagnosisAction
```

## 配置

### 环境变量

- `CHEN1110_XPU_TIMER_PORT`: XPUTimer 服务端口（默认：无）
- `CHEN1110_MASTER_ADDR`: Master 服务地址（用于上报数据）
- `CHEN1110_NODE_ID`: 节点 ID
- `CHEN1110_NODE_TYPE`: 节点类型
- `CHEN1110_MONITOR_ENABLED`: 是否启用监控（默认：false）

## 开发

### 运行测试

```bash
pytest tests/
```

### 代码格式化

```bash
black chen1110/
```

### 代码检查

```bash
flake8 chen1110/
mypy chen1110/
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

Apache License 2.0

## 致谢

本项目参考了 [DLRover](https://github.com/intelligent-machine-learning/dlrover) 的设计架构。

