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

"""Metric collector for XPUTimer performance metrics."""

import logging

import requests

from arobust.agent.data_collector.data_collector import DataCollector
from arobust.common.constants import DiagnosisDataType, EnvConfigKey
from arobust.common.diagnosis_data import WorkerTrainingMetric
from arobust.common.utils import get_env, get_node_id, get_node_rank, get_node_type, is_port_in_use

logger = logging.getLogger(__name__)


class MetricCollector(DataCollector):
    """
    MetricCollector collects GPU metrics from xpu-timer.
    """

    def __init__(self):
        super().__init__()
        self._metric_port = get_env(EnvConfigKey.XPU_TIMER_PORT)
        if self._metric_port:
            self._metric_endpoint = (
                "http://127.0.0.1:" + self._metric_port + "/metrics"
            )
        else:
            self._metric_endpoint = None
        self._client = None  # Will be set when needed

    def collect_data(self) -> str:
        """
        Collect metrics from XPUTimer.

        Returns:
            Preprocessed metric string.
        """
        if not self.is_enabled():
            return ""

        try:
            response = requests.get(self._metric_endpoint, timeout=5)
            response.raise_for_status()

            # Data preprocessing
            return self._preprocess_metrics(response.text)
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Error fetching metrics from xpu-timer: {self._metric_endpoint}, error: {e}"
            )
            return ""

    def _preprocess_metrics(self, metric_str: str) -> str:
        """
        Preprocess metrics by removing comments and exposer lines.

        Args:
            metric_str: Raw metric string.

        Returns:
            Preprocessed metric string.
        """
        try:
            metric_list = [
                line
                for line in metric_str.splitlines()
                if not line.startswith("#") and not line.startswith("exposer")
            ]
            return "\n".join(metric_list)
        except Exception as e:
            logger.warning(f"Error preprocessing metrics from xpu-timer: {e}")
            return ""

    def is_enabled(self) -> bool:
        """
        Check if XPUTimer metric collection is enabled.

        Returns:
            True if XPUTimer is available and port is in use.
        """
        return self._metric_endpoint is not None and is_port_in_use(
            self._metric_port
        )

    def store_data(self, data: object):
        """
        Store metric data by reporting to master.

        Args:
            data: Metric string to store.
        """
        if not isinstance(data, str):
            logger.warning("The data is not of type string")
            return

        # Create metric object
        agent_xpu_metric = WorkerTrainingMetric(
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=data,
            node_id=get_node_id(),
            node_type=get_node_type(),
            node_rank=get_node_rank(),
        )

        # In a real implementation, this would report to master
        # For now, just log it
        logger.info(f"Collected XPU metrics: {len(data)} bytes")

