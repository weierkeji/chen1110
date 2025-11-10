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

"""Utility functions."""

import os
import socket
from typing import Optional


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable."""
    return os.getenv(key, default)


def get_node_id() -> int:
    """Get node ID from environment."""
    from chen1110.common.constants import EnvConfigKey

    node_id_str = get_env(EnvConfigKey.NODE_ID, "-1")
    try:
        return int(node_id_str)
    except ValueError:
        return -1


def get_node_type() -> str:
    """Get node type from environment."""
    from chen1110.common.constants import EnvConfigKey

    return get_env(EnvConfigKey.NODE_TYPE, "worker")


def get_node_rank() -> int:
    """Get node rank from environment."""
    from chen1110.common.constants import EnvConfigKey

    node_rank_str = get_env(EnvConfigKey.NODE_RANK, "-1")
    try:
        return int(node_rank_str)
    except ValueError:
        return -1


def get_local_rank() -> int:
    """Get local rank from environment."""
    from chen1110.common.constants import EnvConfigKey

    local_rank_str = get_env(EnvConfigKey.LOCAL_RANK, "0")
    try:
        return int(local_rank_str)
    except ValueError:
        return 0


def is_port_in_use(port: str) -> bool:
    """Check if a port is in use."""
    try:
        port_num = int(port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port_num)) == 0
    except Exception:
        return False


def find_free_port() -> int:
    """Find a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class Singleton:
    """Singleton pattern implementation."""

    _instances = {}
    _instance_lock = {}

    @classmethod
    def singleton_instance(cls, *args, **kwargs):
        """Get or create singleton instance."""
        import threading

        if cls not in cls._instances:
            if cls not in cls._instance_lock:
                cls._instance_lock[cls] = threading.Lock()
            with cls._instance_lock[cls]:
                if cls not in cls._instances:
                    cls._instances[cls] = cls(*args, **kwargs)
        return cls._instances[cls]

