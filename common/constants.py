# Copyright 2025 chen1110. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants for diagnosis and monitoring."""

from enum import Enum, auto


class DiagnosisConstant:
    """Constants for diagnosis system."""

    # Time intervals
    MIN_DIAGNOSIS_INTERVAL = 30  # seconds
    AGENT_PERIODICALLY_REPORT_INTERVAL_SECS = 30  # seconds
    MIN_DATA_COLLECT_INTERVAL = 30  # seconds

    # Instance identifiers
    LOCAL_INSTANCE = "local"


class DiagnosisErrorConstant:
    """Error types for diagnosis."""

    NODE_FAILED = "NODE_FAILED"
    RESOURCE_COLLECT_ERROR = "RESOURCE_COLLECT_ERROR"
    TRAINING_HANG = "TRAINING_HANG"
    GPU_ERROR = "GPU_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    OOM_ERROR = "OOM_ERROR"


class DiagnosisDataType(Enum):
    """Data types for diagnosis."""

    RESOURCE_STATS = auto()
    XPU_TIMER_METRIC = auto()
    TRAINING_LOG = auto()
    STACK_TRACE = auto()
    TRAINING_METRIC = auto()


class DiagnosisActionType(Enum):
    """Action types for diagnosis."""

    NO_ACTION = auto()
    RESTART_WORKER = auto()
    RELAUNCH_WORKER = auto()
    RESTART_NODE = auto()
    STOP_JOB = auto()


class EnvConfigKey:
    """Environment variable keys."""

    # XPUTimer configuration
    XPU_TIMER_PORT = "CHEN1110_XPU_TIMER_PORT"

    # Master configuration
    MASTER_ADDR = "CHEN1110_MASTER_ADDR"

    # Node configuration
    NODE_ID = "CHEN1110_NODE_ID"
    NODE_TYPE = "CHEN1110_NODE_TYPE"
    NODE_RANK = "CHEN1110_NODE_RANK"
    NODE_IP = "NODE_IP"

    # Monitoring configuration
    MONITOR_ENABLED = "CHEN1110_MONITOR_ENABLED"

    # Training configuration
    LOCAL_RANK = "LOCAL_RANK"
    WORLD_SIZE = "WORLD_SIZE"
    RANK = "RANK"


class Accelerators:
    """Accelerator types."""

    NVIDIA_GPU = "NVIDIA_GPU"
    ASCEND_NPU = "ASCEND_NPU"
    AMD_GPU = "AMD_GPU"


class TrainingExceptionLevel(Enum):
    """Training exception severity levels."""

    PROCESS_ERROR = auto()
    NODE_ERROR = auto()
    JOB_ERROR = auto()

