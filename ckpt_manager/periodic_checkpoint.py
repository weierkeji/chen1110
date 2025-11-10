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

"""Periodic checkpoint manager."""

import logging
import os
from typing import Any, Dict

import torch

from arobust.ckpt_manager.latest_checkpoint import LatestCheckpointManager

logger = logging.getLogger(__name__)


class PeriodicCheckpointManager(LatestCheckpointManager):
    """
    Manages periodic checkpoints with configurable save interval.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int = 1000,
        max_checkpoints: int = 5,
    ):
        """
        Initialize periodic checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
            save_interval: Save checkpoint every N steps.
            max_checkpoints: Maximum number of checkpoints to keep.
        """
        super().__init__(checkpoint_dir, max_checkpoints)
        self._save_interval = save_interval
        self._last_saved_step = -1

    def should_save(self, step: int) -> bool:
        """
        Check if should save checkpoint at this step.

        Args:
            step: Current training step.

        Returns:
            True if should save.
        """
        if step == 0:
            return False

        if step - self._last_saved_step >= self._save_interval:
            return True

        return False

    def save_if_needed(self, state_dict: Dict[str, Any], step: int) -> bool:
        """
        Save checkpoint if needed based on save_interval.

        Args:
            state_dict: State dictionary to save.
            step: Current training step.

        Returns:
            True if checkpoint was saved.
        """
        if self.should_save(step):
            checkpoint_path = self.save(state_dict, step)
            if checkpoint_path:
                self._last_saved_step = step
                return True

        return False

    def save(self, state_dict: Dict[str, Any], step: int) -> str:
        """
        Save checkpoint.

        Args:
            state_dict: State dictionary to save.
            step: Current training step.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = super().save(state_dict, step)
        if checkpoint_path:
            self._last_saved_step = step
        return checkpoint_path

