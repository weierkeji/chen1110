# Copyright 2025 arobust. All rights reserved.
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

"""Resource collector for CPU, memory, and GPU usage."""

from arobust.agent.data_collector.data_collector import DataCollector


class ResourceCollector(DataCollector):
    """
    ResourceCollector collects the resource data.
    """

    def __init__(self):
        super().__init__()
        # Import here to avoid circular dependency
        from arobust.agent.monitor.resource import ResourceMonitor

        self._monitor = ResourceMonitor.singleton_instance()

    def collect_data(self) -> object:
        """
        Collect resource data by triggering resource monitor report.

        Returns:
            None (data is reported directly to master).
        """
        self._monitor.report_resource()
        return None

    def is_enabled(self) -> bool:
        """
        Check if resource collection is enabled.

        Returns:
            True if enabled.
        """
        return True

