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

"""Data manager for diagnosis system."""

import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional

from chen1110.common.diagnosis_data import WorkerTrainingMetric

logger = logging.getLogger(__name__)


class DiagnosisDataManager:
    """
    Manages diagnosis data with time-based expiration.
    """

    def __init__(self, data_retention_time: int = 600):
        """
        Initialize data manager.

        Args:
            data_retention_time: How long to retain data in seconds (default 600s).
        """
        self._data_retention_time = data_retention_time
        self._data_store: Dict[str, Deque[WorkerTrainingMetric]] = {}
        self._lock = threading.Lock()

    def store_data(self, data: WorkerTrainingMetric):
        """
        Store diagnosis data.

        Args:
            data: WorkerTrainingMetric object to store.
        """
        with self._lock:
            data_type = data.data_type.name
            if data_type not in self._data_store:
                self._data_store[data_type] = deque()

            # Add timestamp if not set
            if data.timestamp == 0:
                data.timestamp = int(time.time())

            self._data_store[data_type].append(data)

            # Clean old data
            self._clean_old_data(data_type)

    def get_data(
        self,
        data_type: str,
        node_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[WorkerTrainingMetric]:
        """
        Retrieve diagnosis data.

        Args:
            data_type: Type of data to retrieve.
            node_id: Optional node ID filter.
            limit: Maximum number of records to return.

        Returns:
            List of WorkerTrainingMetric objects.
        """
        with self._lock:
            if data_type not in self._data_store:
                return []

            data_list = list(self._data_store[data_type])

            # Filter by node_id if specified
            if node_id is not None:
                data_list = [d for d in data_list if d.node_id == node_id]

            # Return most recent records up to limit
            return data_list[-limit:]

    def get_latest_data(
        self, data_type: str, node_id: Optional[int] = None
    ) -> Optional[WorkerTrainingMetric]:
        """
        Get the most recent data entry.

        Args:
            data_type: Type of data to retrieve.
            node_id: Optional node ID filter.

        Returns:
            Most recent WorkerTrainingMetric or None.
        """
        data_list = self.get_data(data_type, node_id, limit=1)
        return data_list[0] if data_list else None

    def _clean_old_data(self, data_type: str):
        """
        Remove data older than retention time.

        Args:
            data_type: Type of data to clean.
        """
        if data_type not in self._data_store:
            return

        current_time = int(time.time())
        cutoff_time = current_time - self._data_retention_time

        # Remove old entries from the front of the deque
        while self._data_store[data_type]:
            if self._data_store[data_type][0].timestamp < cutoff_time:
                self._data_store[data_type].popleft()
            else:
                break

    def clear_data(self, data_type: Optional[str] = None):
        """
        Clear stored data.

        Args:
            data_type: Optional specific data type to clear. If None, clear all.
        """
        with self._lock:
            if data_type is None:
                self._data_store.clear()
            elif data_type in self._data_store:
                self._data_store[data_type].clear()

    def get_data_count(self, data_type: str) -> int:
        """
        Get count of stored data for a type.

        Args:
            data_type: Type of data to count.

        Returns:
            Number of stored entries.
        """
        with self._lock:
            if data_type not in self._data_store:
                return 0
            return len(self._data_store[data_type])

