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

"""Resource monitor for CPU, memory, and GPU usage."""

import logging
import os
import threading
import time

import psutil

from chen1110.common.constants import Accelerators, EnvConfigKey
from chen1110.common.diagnosis_data import GPUStats
from chen1110.common.utils import Singleton, get_env

logger = logging.getLogger(__name__)


def get_process_cpu_percent() -> float:
    """
    Get the CPU percent of the current process.

    Returns:
        CPU usage percentage (0.0 to 1.0).
    """
    try:
        proc_total_percent = 0.0
        for proc in psutil.process_iter(
            ["pid", "ppid", "name", "username", "cmdline"]
        ):
            proc_percent = proc.cpu_percent()
            proc_total_percent += proc_percent

        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = round(proc_total_percent / cpu_count, 2)
    except Exception:
        cpu_percent = 0.0
    return cpu_percent / 100.0


def get_used_memory() -> int:
    """
    Get the used memory of the container/process in MB.

    Returns:
        Used memory in MB.
    """
    mem = psutil.virtual_memory()
    return int(mem.used / 1024 / 1024)


def get_gpu_stats(gpus=None) -> list[GPUStats]:
    """
    Get the used GPU info.

    Args:
        gpus: List of GPU indices to query. If None, query all GPUs.

    Returns:
        List of GPUStats objects.
    """
    if gpus is None:
        gpus = []

    try:
        import pynvml
    except ImportError:
        logger.warning("No pynvml is available, skip getting gpu stats.")
        return []

    try:
        pynvml.nvmlInit()

        if not gpus:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
            except Exception:
                logger.warning("No GPU is available.")
                device_count = 0
            gpus = list(range(device_count))

        gpu_stats: list[GPUStats] = []
        for i in gpus:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total / (1024**2)
            used_memory = memory_info.used / (1024**2)

            # Get GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = utilization.gpu

            gpu_stats.append(
                GPUStats(
                    index=i,
                    total_memory_mb=total_memory,
                    used_memory_mb=used_memory,
                    gpu_utilization=gpu_utilization,
                )
            )
        return gpu_stats
    except Exception as e:
        logger.warning(f"Got unexpected error when getting gpu stats: {e}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


class ResourceMonitor(Singleton):
    """
    The monitor samples the used memory and CPU percent
    and reports the used memory and CPU percent to the master.
    """

    def __init__(self, gpu_type: str = Accelerators.NVIDIA_GPU):
        """
        Initialize resource monitor.

        Args:
            gpu_type: Type of accelerator (NVIDIA_GPU, ASCEND_NPU, etc.).
        """
        self._total_cpu = psutil.cpu_count(logical=True)
        self._gpu_type = gpu_type
        self._gpu_stats: list[GPUStats] = []
        self._master_client = None  # Will be set when integrated with master
        self._stopped = False
        self._monitor_thread = None

        master_addr = get_env(EnvConfigKey.MASTER_ADDR, "")
        if master_addr:
            # The first time called cpu_percent will return a meaningless 0.0
            # value which we are supposed to ignore. So, here we call it at
            # the beginning of monitor and the next value is valid.
            get_process_cpu_percent()

    def start(self):
        """Start the resource monitoring thread."""
        master_addr = get_env(EnvConfigKey.MASTER_ADDR, "")
        if not master_addr:
            logger.info("No master address configured, skipping resource monitor.")
            return

        logger.info("Resource Monitor Initializing ...")

        try:
            self._stopped = False
            self._monitor_thread = threading.Thread(
                target=self._monitor_resource,
                name="monitor_resource",
                daemon=True,
            )
            self._monitor_thread.start()
            if self._monitor_thread.is_alive():
                logger.info("Resource Monitor initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to start the monitor resource thread. Error: {e}"
            )

    def stop(self):
        """Stop the resource monitoring thread."""
        self._stopped = True
        if self._monitor_thread:
            # Wait for thread to finish
            self._monitor_thread.join(timeout=5)

    def report_resource(self):
        """Report resource usage to master."""
        used_mem = get_used_memory()
        cpu_percent = get_process_cpu_percent()

        if self._gpu_type == Accelerators.NVIDIA_GPU:
            self._gpu_stats = get_gpu_stats()
        else:
            # Not supported for other accelerators yet
            pass

        current_cpu = round(cpu_percent * self._total_cpu, 2)

        # In a real implementation, this would report to master client
        # self._master_client.report_used_resource(used_mem, current_cpu, self._gpu_stats)

        logger.debug(
            f"Report Resource CPU: {current_cpu}, Memory: {used_mem}, GPU: {len(self._gpu_stats)} devices"
        )

    def _monitor_resource(self):
        """Background thread for monitoring resources."""
        logger.info("Start to monitor resource usage")
        while not self._stopped:
            try:
                self.report_resource()
            except Exception as e:
                logger.debug(f"report resource error: {e}")
            time.sleep(15)  # Report every 15 seconds

