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

"""Log collector for training logs."""

import logging
import os
from typing import Optional

from chen1110.agent.data_collector.data_collector import DataCollector
from chen1110.common.constants import DiagnosisDataType
from chen1110.common.diagnosis_data import WorkerTrainingMetric
from chen1110.common.utils import get_node_id, get_node_rank, get_node_type

logger = logging.getLogger(__name__)


class LogCollector(DataCollector):
    """
    LogCollector collects training log file content.
    Used for error analysis and diagnosis.
    """

    def __init__(self, log_file_path: Optional[str] = None, max_lines: int = 1000):
        """
        Initialize log collector.

        Args:
            log_file_path: Path to the training log file.
            max_lines: Maximum number of lines to collect from the end of the file.
        """
        super().__init__()
        self._log_file_path = log_file_path
        self._max_lines = max_lines
        self._last_position = 0

    def collect_data(self) -> str:
        """
        Collect log file content.

        Returns:
            Log content string.
        """
        if not self.is_enabled():
            return ""

        try:
            with open(self._log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                # Read from last position
                f.seek(self._last_position)
                new_content = f.read()
                self._last_position = f.tell()

                # If content is too long, only keep last N lines
                lines = new_content.splitlines()
                if len(lines) > self._max_lines:
                    lines = lines[-self._max_lines :]
                    new_content = "\n".join(lines)

                return new_content
        except FileNotFoundError:
            logger.warning(f"Log file not found: {self._log_file_path}")
            return ""
        except Exception as e:
            logger.warning(f"Error reading log file: {e}")
            return ""

    def is_enabled(self) -> bool:
        """
        Check if log collection is enabled.

        Returns:
            True if log file exists.
        """
        return (
            self._log_file_path is not None
            and os.path.exists(self._log_file_path)
        )

    def store_data(self, data: object):
        """
        Store log data by reporting to master.

        Args:
            data: Log content string.
        """
        if not isinstance(data, str):
            logger.warning("The data is not of type string")
            return

        if not data:
            # No new log content
            return

        # Create metric object
        agent_log_metric = WorkerTrainingMetric(
            data_type=DiagnosisDataType.TRAINING_LOG,
            data_content=data,
            node_id=get_node_id(),
            node_type=get_node_type(),
            node_rank=get_node_rank(),
        )

        # In a real implementation, this would report to master
        logger.info(f"Collected log content: {len(data)} characters")

    def reset_position(self):
        """Reset the file read position to 0."""
        self._last_position = 0

