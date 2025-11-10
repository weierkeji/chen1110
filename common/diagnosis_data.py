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

"""Data structures for diagnosis system."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from chen1110.common.constants import DiagnosisActionType, DiagnosisDataType


@dataclass
class GPUStats:
    """GPU statistics."""

    index: int
    total_memory_mb: float
    used_memory_mb: float
    gpu_utilization: float


@dataclass
class ResourceStats:
    """Resource statistics."""

    cpu_percent: float
    memory_mb: int
    gpu_stats: list[GPUStats] = field(default_factory=list)
    timestamp: int = 0


@dataclass
class WorkerTrainingMetric:
    """Training metrics from worker."""

    data_type: DiagnosisDataType
    data_content: str
    node_id: int = -1
    node_type: str = ""
    node_rank: int = -1
    timestamp: int = 0

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(
            {
                "data_type": self.data_type.name,
                "data_content": self.data_content,
                "node_id": self.node_id,
                "node_type": self.node_type,
                "node_rank": self.node_rank,
                "timestamp": self.timestamp,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "WorkerTrainingMetric":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            data_type=DiagnosisDataType[data["data_type"]],
            data_content=data["data_content"],
            node_id=data.get("node_id", -1),
            node_type=data.get("node_type", ""),
            node_rank=data.get("node_rank", -1),
            timestamp=data.get("timestamp", 0),
        )


@dataclass
class DiagnosisObservation:
    """Observation from diagnosis."""

    observation: str = ""
    is_anomaly: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosisAction:
    """Action to take based on diagnosis."""

    action_type: DiagnosisActionType
    node_id: Optional[int] = None
    node_type: Optional[str] = None
    instance: str = ""
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(
            {
                "action_type": self.action_type.name,
                "node_id": self.node_id,
                "node_type": self.node_type,
                "instance": self.instance,
                "reason": self.reason,
                "details": self.details,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DiagnosisAction":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            action_type=DiagnosisActionType[data["action_type"]],
            node_id=data.get("node_id"),
            node_type=data.get("node_type"),
            instance=data.get("instance", ""),
            reason=data.get("reason", ""),
            details=data.get("details", {}),
        )


@dataclass
class NoAction(DiagnosisAction):
    """No action needed."""

    def __init__(self):
        super().__init__(action_type=DiagnosisActionType.NO_ACTION)


@dataclass
class NodeAction(DiagnosisAction):
    """Action on a node."""

    def __init__(
        self,
        node_id: int,
        node_type: str,
        instance: str,
        action_type: DiagnosisActionType,
        reason: str = "",
    ):
        super().__init__(
            action_type=action_type,
            node_id=node_id,
            node_type=node_type,
            instance=instance,
            reason=reason,
        )


@dataclass
class ProcessError:
    """Process error information."""

    local_rank: int
    exitcode: int
    message: str
    timestamp: str

