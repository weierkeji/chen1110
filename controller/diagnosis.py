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

"""Diagnosis agent for fault tolerance."""

import logging
import threading
import time
from typing import Dict, Optional

from chen1110.agent.data_collector import (
    DataCollector,
    LogCollector,
    MetricCollector,
    ResourceCollector,
    StackCollector,
)
from chen1110.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
    DiagnosisErrorConstant,
)
from chen1110.common.diagnosis_data import DiagnosisAction, NoAction, NodeAction
from chen1110.common.utils import Singleton, get_node_id, get_node_type

logger = logging.getLogger(__name__)


class DiagnosisAgent(Singleton):
    """
    DiagnosisAgent manages diagnosis data collection and fault detection.
    It is the top-level orchestrator that integrates data collectors,
    monitors, and diagnosticians.
    """

    def __init__(
        self,
        training_log_file: str = "",
        errors: str = "",
        node_rank: int = -1,
        local_world_size: int = 0,
    ):
        """
        Initialize diagnosis agent.

        Args:
            training_log_file: Path to training log file.
            errors: Error messages from training.
            node_rank: Rank of current node.
            local_world_size: Number of processes on this node.
        """
        self._training_log_file = training_log_file
        self._errors = errors
        self._node_rank = node_rank
        self._local_world_size = local_world_size
        self._stopped = False

        # Data collectors
        self._periodical_collectors: Dict[DataCollector, int] = {}
        self._lock = threading.Lock()

        # Threads
        self._report_thread = None
        self._collector_threads = []

        logger.info(
            f"Initializing diagnosis agent with\n"
            f"training_log_file:    {self._training_log_file}\n"
            f"errors:               {self._errors}\n"
            f"node_rank:            {self._node_rank}"
        )

    def register_periodical_data_collector(
        self, collector: DataCollector, time_interval: int
    ):
        """
        Register a data collector to run periodically.

        Args:
            collector: DataCollector instance.
            time_interval: Collection interval in seconds.
        """
        with self._lock:
            if time_interval < DiagnosisConstant.MIN_DATA_COLLECT_INTERVAL:
                time_interval = DiagnosisConstant.MIN_DATA_COLLECT_INTERVAL

            self._periodical_collectors[collector] = time_interval
            logger.info(
                f"Registered periodic collector {collector.__class__.__name__} "
                f"with interval {time_interval}s"
            )

    def start(self):
        """Start the diagnosis agent."""
        self._setup_default_collectors()
        self._start_data_collection()
        self._start_periodic_report()

        self._stopped = False
        logger.info("Diagnosis agent started successfully")

    def stop(self):
        """Stop the diagnosis agent."""
        self._stopped = True

        # Wait for threads to finish
        if self._report_thread:
            self._report_thread.join(timeout=5)

        for thread in self._collector_threads:
            thread.join(timeout=5)

        logger.info("Diagnosis agent stopped")

    def _setup_default_collectors(self):
        """Setup default data collectors."""
        # Register resource collector (30s interval)
        resource_collector = ResourceCollector()
        self.register_periodical_data_collector(resource_collector, 30)

        # Register metric collector (60s interval)
        metric_collector = MetricCollector()
        self.register_periodical_data_collector(metric_collector, 60)

        # Register log collector if log file is specified
        if self._training_log_file:
            log_collector = LogCollector(self._training_log_file)
            self.register_periodical_data_collector(log_collector, 60)

        # Register stack collector (120s interval)
        stack_collector = StackCollector()
        self.register_periodical_data_collector(stack_collector, 120)

    def _start_data_collection(self):
        """Start periodic data collection threads."""
        with self._lock:
            logger.info(
                f"Starting {len(self._periodical_collectors)} periodic data collectors"
            )

            for collector, time_interval in self._periodical_collectors.items():
                try:
                    thread_name = (
                        f"periodical_collector_{collector.__class__.__name__}"
                    )
                    thread = threading.Thread(
                        target=self._start_periodical_collector,
                        name=thread_name,
                        args=(collector, time_interval),
                        daemon=True,
                    )
                    thread.start()
                    self._collector_threads.append(thread)

                    if thread.is_alive():
                        logger.info(f"{thread_name} initialized successfully")
                    else:
                        logger.error(f"{thread_name} failed to start")
                except Exception as e:
                    logger.error(
                        f"Failed to start collector thread: {e}"
                    )

    def _start_periodical_collector(
        self, collector: DataCollector, time_interval: int
    ):
        """
        Run a data collector periodically.

        Args:
            collector: DataCollector instance.
            time_interval: Collection interval in seconds.
        """
        name = collector.__class__.__name__
        logger.info(f"Started periodic collector: {name}")

        while not self._stopped:
            time.sleep(time_interval)
            try:
                if collector.is_enabled():
                    data = collector.collect_data()
                    if data is not None:
                        collector.store_data(data)
            except Exception as e:
                logger.error(f"Error in collector {name}: {e}")

    def _start_periodic_report(self):
        """Start periodic heartbeat reporting thread."""
        self._report_thread = threading.Thread(
            target=self._periodically_report,
            name="periodically_reporter",
            daemon=True,
        )
        self._report_thread.start()

    def _periodically_report(self):
        """Periodic reporting loop."""
        logger.info("Start diagnosis agent periodically reporter.")
        while not self._stopped:
            try:
                self.send_heartbeat()
            except Exception as e:
                logger.warning(f"Error in periodic reporting: {e}")

            time.sleep(DiagnosisConstant.AGENT_PERIODICALLY_REPORT_INTERVAL_SECS)

    def send_heartbeat(self):
        """Send heartbeat to master."""
        try:
            ts = int(time.time())
            # In a real implementation, this would communicate with master
            # action = self._client.report_heart_beat(ts)
            logger.debug(f"Sent heartbeat at timestamp: {ts}")
        except Exception as e:
            logger.warning(f"Failed to send heartbeat: {e}")

    def diagnose_training_failure(
        self, failures: Dict = None, restart_count: int = 0
    ) -> DiagnosisAction:
        """
        Diagnose training failure and determine action.

        Args:
            failures: Dictionary of failures.
            restart_count: Number of restarts attempted.

        Returns:
            DiagnosisAction to take.
        """
        if failures is None:
            failures = {}

        logger.info(
            f"Diagnosing training failure: {len(failures)} failures, "
            f"restart count: {restart_count}"
        )

        # Simple diagnosis logic: if we have restarts remaining, restart worker
        # otherwise, relaunch node
        max_restarts = 3  # This should come from configuration

        if restart_count < max_restarts:
            logger.info(
                f"Worker failure detected, {max_restarts - restart_count} "
                f"restart attempts remaining"
            )
            return NodeAction(
                node_id=get_node_id(),
                node_type=get_node_type(),
                instance=DiagnosisConstant.LOCAL_INSTANCE,
                action_type=DiagnosisActionType.RESTART_WORKER,
                reason="Training process failed, restarting worker",
            )
        else:
            logger.info(
                f"Max restarts ({max_restarts}) reached, relaunching node"
            )
            return NodeAction(
                node_id=get_node_id(),
                node_type=get_node_type(),
                instance=DiagnosisConstant.LOCAL_INSTANCE,
                action_type=DiagnosisActionType.RELAUNCH_WORKER,
                reason="Max restart attempts reached, relaunching node",
            )

    def update_config(
        self,
        training_log_file: str = "",
        errors: str = "",
        node_rank: int = -1,
    ):
        """
        Update configuration.

        Args:
            training_log_file: Path to training log file.
            errors: Error messages.
            node_rank: Node rank.
        """
        if training_log_file:
            self._training_log_file = training_log_file
            logger.info(f"Updated training_log_file: {training_log_file}")

        if errors:
            self._errors = errors
            logger.info(f"Updated errors: {errors}")

        if node_rank >= 0:
            self._node_rank = node_rank
            logger.info(f"Updated node_rank: {node_rank}")

