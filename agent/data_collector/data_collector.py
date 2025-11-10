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

"""Base class for data collectors."""

from abc import ABCMeta, abstractmethod


class DataCollector(metaclass=ABCMeta):
    """
    DataCollector collects certain type of data and report to master.
    Those data is used to diagnose the faults of training.
    """

    def __init__(self):
        pass

    @abstractmethod
    def collect_data(self) -> object:
        """
        The implementation of data collector.

        Returns:
            Collected data object.
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Whether the collector is enabled.

        Returns:
            True if enabled, False otherwise.
        """
        return True

    def store_data(self, data: object):
        """
        Store the collected data.

        Args:
            data: The data to store.
        """
        pass

