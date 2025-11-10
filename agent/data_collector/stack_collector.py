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

"""Stack collector for training process stack traces."""

import logging
import sys
import traceback
from typing import Dict, List

from chen1110.agent.data_collector.data_collector import DataCollector
from chen1110.common.constants import DiagnosisDataType
from chen1110.common.diagnosis_data import WorkerTrainingMetric
from chen1110.common.utils import get_node_id, get_node_rank, get_node_type

logger = logging.getLogger(__name__)


class StackCollector(DataCollector):
    """
    StackCollector collects stack traces of training processes.
    Used for diagnosing hangs and deadlocks.
    """

    def __init__(self):
        super().__init__()
        self._enabled = True

    def collect_data(self) -> Dict[int, List[str]]:
        """
        Collect stack traces of all threads.

        Returns:
            Dictionary mapping thread ID to stack frames.
        """
        if not self.is_enabled():
            return {}

        try:
            import threading

            stacks = {}
            for thread_id, frame in sys._current_frames().items():
                # Get thread name
                thread_name = "unknown"
                for thread in threading.enumerate():
                    if thread.ident == thread_id:
                        thread_name = thread.name
                        break

                # Extract stack frames
                stack_frames = traceback.format_stack(frame)
                stacks[thread_id] = {
                    "name": thread_name,
                    "frames": stack_frames,
                }

            return stacks
        except Exception as e:
            logger.warning(f"Error collecting stack traces: {e}")
            return {}

    def is_enabled(self) -> bool:
        """
        Check if stack collection is enabled.

        Returns:
            True if enabled.
        """
        return self._enabled

    def store_data(self, data: object):
        """
        Store stack trace data by reporting to master.

        Args:
            data: Stack trace dictionary.
        """
        if not isinstance(data, dict):
            logger.warning("The data is not of type dict")
            return

        # Convert stack data to string
        stack_str = self._format_stacks(data)

        # Create metric object
        agent_stack_metric = WorkerTrainingMetric(
            data_type=DiagnosisDataType.STACK_TRACE,
            data_content=stack_str,
            node_id=get_node_id(),
            node_type=get_node_type(),
            node_rank=get_node_rank(),
        )

        # In a real implementation, this would report to master
        logger.info(f"Collected stack traces for {len(data)} threads")

    def _format_stacks(self, stacks: Dict) -> str:
        """
        Format stack traces for storage.

        Args:
            stacks: Dictionary of stack traces.

        Returns:
            Formatted string.
        """
        lines = []
        for thread_id, thread_data in stacks.items():
            lines.append(f"Thread {thread_id} ({thread_data['name']}):")
            for frame in thread_data["frames"]:
                lines.append(f"  {frame}")
            lines.append("")
        return "\n".join(lines)

