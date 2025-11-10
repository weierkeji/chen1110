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

"""Training monitor for tracking training progress."""

import json
import logging
import os
import threading
import time

from arobust.agent.monitor.resource import ResourceMonitor
from arobust.common.constants import Accelerators, EnvConfigKey
from arobust.common.utils import Singleton, get_env, get_node_rank

logger = logging.getLogger(__name__)


class TrainingMonitor(Singleton):
    """
    TrainingMonitor monitors training progress and reports to master.
    """

    def __init__(
        self, metrics_path: str, device_type: str = Accelerators.NVIDIA_GPU
    ):
        """
        Initialize training monitor.

        Args:
            metrics_path: Path to the training metrics file.
            device_type: Type of accelerator device.
        """
        self._resource_monitor = ResourceMonitor.singleton_instance(device_type)
        self._last_timestamp = 0
        self._start_time = 0
        self._master_client = None  # Will be set when integrated with master
        self._group_rank = get_node_rank()
        self._metrics_path = metrics_path
        self._stopped = False
        self._monitor_thread = None

        # Remove old metrics file if exists
        if os.path.exists(metrics_path):
            try:
                os.remove(metrics_path)
            except Exception as e:
                logger.warning(f"Failed to remove old metrics file: {e}")

    def start(self):
        """Start the training monitoring thread."""
        monitor_enabled = get_env(EnvConfigKey.MONITOR_ENABLED, "false")
        if monitor_enabled != "true":
            logger.info(
                f"Skip starting monitor for {EnvConfigKey.MONITOR_ENABLED} disabled."
            )
            return

        logger.info("Training Monitor Initializing ...")

        try:
            self._stopped = False
            self._monitor_thread = threading.Thread(
                target=self._periodically_report,
                name="node_reporter",
                daemon=True,
            )
            self._monitor_thread.start()
            if self._monitor_thread.is_alive():
                logger.info("Training Monitor initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to start the training monitor thread. Error: {e}"
            )

    def stop(self):
        """Stop the training monitoring thread."""
        self._stopped = True
        self._resource_monitor.stop()
        if self._monitor_thread:
            # Wait for thread to finish
            self._monitor_thread.join(timeout=5)

    def report_step(self):
        """Report training step progress to master."""
        if self._group_rank != 0:
            # Only rank 0 reports
            return

        try:
            if not os.path.exists(self._metrics_path):
                return

            with open(self._metrics_path, "r") as f:
                record = json.load(f)
                step = record.get("step", 0)
                timestamp = record.get("timestamp", 0)

            if step > 0 and timestamp - self._last_timestamp > 15:
                self._last_timestamp = timestamp

                # In a real implementation, this would report to master client
                # self._master_client.report_global_step(step, self._last_timestamp)

                logger.debug(f"Report global step: {step} at timestamp: {timestamp}")
        except Exception as e:
            logger.warning(f"Error reporting step: {e}")

    def _periodically_report(self):
        """Background thread for periodic reporting."""
        logger.info("Start training agent reporter.")
        while not self._stopped:
            if self._group_rank == 0:
                self.report_step()
            time.sleep(15)  # Report every 15 seconds


def write_training_metrics(metrics_path: str, step: int, timestamp: int = None):
    """
    Helper function to write training metrics to file.

    Args:
        metrics_path: Path to metrics file.
        step: Current training step.
        timestamp: Current timestamp (defaults to current time).
    """
    if timestamp is None:
        timestamp = int(time.time())

    try:
        with open(metrics_path, "w") as f:
            json.dump({"step": step, "timestamp": timestamp}, f)
    except Exception as e:
        logger.warning(f"Failed to write training metrics: {e}")

