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

"""Latest checkpoint manager."""

import logging
import os
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class LatestCheckpointManager:
    """
    Manages the latest training checkpoint with fast save/load.
    """

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        """
        Initialize latest checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints.
            max_checkpoints: Maximum number of checkpoints to keep.
        """
        self._checkpoint_dir = checkpoint_dir
        self._max_checkpoints = max_checkpoints
        self._latest_checkpoint_path = None

        # Create checkpoint directory if not exists
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, state_dict: Dict[str, Any], step: int) -> str:
        """
        Save checkpoint.

        Args:
            state_dict: State dictionary to save.
            step: Current training step.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = os.path.join(
            self._checkpoint_dir, f"checkpoint_step_{step}.pt"
        )

        try:
            torch.save(state_dict, checkpoint_path)
            self._latest_checkpoint_path = checkpoint_path
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Clean old checkpoints
            self._cleanup_old_checkpoints()

            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return ""

    def load(self, checkpoint_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint. If None, load latest.

        Returns:
            State dictionary or None if load fails.
        """
        if checkpoint_path is None:
            checkpoint_path = self._latest_checkpoint_path

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """
        Get path to the latest checkpoint.

        Returns:
            Path to latest checkpoint or None.
        """
        if self._latest_checkpoint_path and os.path.exists(
            self._latest_checkpoint_path
        ):
            return self._latest_checkpoint_path

        # Find latest checkpoint in directory
        checkpoints = self._list_checkpoints()
        if checkpoints:
            return checkpoints[-1]

        return None

    def _list_checkpoints(self) -> list[str]:
        """
        List all checkpoints in directory sorted by step.

        Returns:
            List of checkpoint paths.
        """
        if not os.path.exists(self._checkpoint_dir):
            return []

        checkpoints = []
        for filename in os.listdir(self._checkpoint_dir):
            if filename.startswith("checkpoint_step_") and filename.endswith(".pt"):
                checkpoint_path = os.path.join(self._checkpoint_dir, filename)
                checkpoints.append(checkpoint_path)

        # Sort by step number
        checkpoints.sort(
            key=lambda x: int(x.split("_step_")[-1].split(".pt")[0])
        )
        return checkpoints

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding max_checkpoints."""
        checkpoints = self._list_checkpoints()

        if len(checkpoints) > self._max_checkpoints:
            # Remove oldest checkpoints
            to_remove = checkpoints[: -self._max_checkpoints]
            for checkpoint_path in to_remove:
                try:
                    os.remove(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint: {e}")

