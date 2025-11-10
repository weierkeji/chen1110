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

"""Reference LogP checkpoint manager for PPO."""

import logging
import os
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class RefLogPCheckpointManager:
    """
    Manages Reference model LogP checkpoints for PPO training.
    Used for KL divergence calculation between actor and reference model.
    """

    def __init__(self, checkpoint_dir: str):
        """
        Initialize RefLogP checkpoint manager.

        Args:
            checkpoint_dir: Directory to store reference LogP checkpoints.
        """
        self._checkpoint_dir = os.path.join(checkpoint_dir, "ref_logp")
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def save_ref_logp(
        self, logp_data: torch.Tensor, episode: int, step: int
    ) -> str:
        """
        Save reference log probabilities.

        Args:
            logp_data: Log probability tensor.
            episode: Current episode number.
            step: Current step number.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = os.path.join(
            self._checkpoint_dir, f"ref_logp_ep{episode}_step{step}.pt"
        )

        try:
            torch.save(
                {
                    "logp": logp_data,
                    "episode": episode,
                    "step": step,
                },
                checkpoint_path,
            )
            logger.info(f"Saved reference LogP to {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save reference LogP: {e}")
            return ""

    def load_ref_logp(
        self, episode: int, step: int
    ) -> Optional[torch.Tensor]:
        """
        Load reference log probabilities.

        Args:
            episode: Episode number.
            step: Step number.

        Returns:
            Log probability tensor or None if not found.
        """
        checkpoint_path = os.path.join(
            self._checkpoint_dir, f"ref_logp_ep{episode}_step{step}.pt"
        )

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Reference LogP not found: {checkpoint_path}")
            return None

        try:
            data = torch.load(checkpoint_path, map_location="cpu")
            logger.info(f"Loaded reference LogP from {checkpoint_path}")
            return data["logp"]
        except Exception as e:
            logger.error(f"Failed to load reference LogP: {e}")
            return None

    def get_latest_ref_logp(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest reference LogP checkpoint.

        Returns:
            Dictionary with logp, episode, and step or None.
        """
        checkpoints = self._list_ref_logp_checkpoints()
        if not checkpoints:
            return None

        latest_path = checkpoints[-1]
        try:
            data = torch.load(latest_path, map_location="cpu")
            logger.info(f"Loaded latest reference LogP from {latest_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load latest reference LogP: {e}")
            return None

    def _list_ref_logp_checkpoints(self) -> list[str]:
        """
        List all reference LogP checkpoints sorted by episode and step.

        Returns:
            List of checkpoint paths.
        """
        if not os.path.exists(self._checkpoint_dir):
            return []

        checkpoints = []
        for filename in os.listdir(self._checkpoint_dir):
            if filename.startswith("ref_logp_") and filename.endswith(".pt"):
                checkpoint_path = os.path.join(self._checkpoint_dir, filename)
                checkpoints.append(checkpoint_path)

        # Sort by episode and step
        def get_episode_step(path):
            filename = os.path.basename(path)
            parts = filename.replace("ref_logp_ep", "").replace(".pt", "").split("_step")
            return (int(parts[0]), int(parts[1]))

        checkpoints.sort(key=get_episode_step)
        return checkpoints

    def clear_old_checkpoints(self, keep_last_n: int = 3):
        """
        Remove old checkpoints, keeping only the last N.

        Args:
            keep_last_n: Number of recent checkpoints to keep.
        """
        checkpoints = self._list_ref_logp_checkpoints()

        if len(checkpoints) > keep_last_n:
            to_remove = checkpoints[:-keep_last_n]
            for checkpoint_path in to_remove:
                try:
                    os.remove(checkpoint_path)
                    logger.info(f"Removed old reference LogP: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint: {e}")

