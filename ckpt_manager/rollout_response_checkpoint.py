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

"""Rollout response checkpoint manager."""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RolloutResponseCheckpointManager:
    """
    Manages rollout response checkpoints for experience replay.
    Stores generated rollout responses and rewards for RL training.
    """

    def __init__(self, checkpoint_dir: str):
        """
        Initialize rollout response checkpoint manager.

        Args:
            checkpoint_dir: Directory to store rollout checkpoints.
        """
        self._checkpoint_dir = os.path.join(checkpoint_dir, "rollout_responses")
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    def save_rollout_batch(
        self,
        responses: List[Dict[str, Any]],
        episode: int,
        batch_id: int,
    ) -> str:
        """
        Save a batch of rollout responses.

        Args:
            responses: List of response dictionaries containing:
                - prompts: Input prompts
                - responses: Generated responses
                - rewards: Computed rewards
                - values: Value estimates (optional)
                - logps: Log probabilities (optional)
            episode: Current episode number.
            batch_id: Batch identifier.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = os.path.join(
            self._checkpoint_dir,
            f"rollout_ep{episode}_batch{batch_id}.pkl",
        )

        try:
            data = {
                "responses": responses,
                "episode": episode,
                "batch_id": batch_id,
                "num_samples": len(responses),
            }

            with open(checkpoint_path, "wb") as f:
                pickle.dump(data, f)

            logger.info(
                f"Saved {len(responses)} rollout responses to {checkpoint_path}"
            )
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save rollout responses: {e}")
            return ""

    def load_rollout_batch(
        self, episode: int, batch_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Load a batch of rollout responses.

        Args:
            episode: Episode number.
            batch_id: Batch identifier.

        Returns:
            Dictionary with responses, episode, batch_id, num_samples or None.
        """
        checkpoint_path = os.path.join(
            self._checkpoint_dir,
            f"rollout_ep{episode}_batch{batch_id}.pkl",
        )

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Rollout batch not found: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)

            logger.info(
                f"Loaded {data['num_samples']} rollout responses from {checkpoint_path}"
            )
            return data
        except Exception as e:
            logger.error(f"Failed to load rollout responses: {e}")
            return None

    def list_rollout_batches(self, episode: Optional[int] = None) -> List[str]:
        """
        List all rollout batch checkpoints.

        Args:
            episode: Optional episode number to filter by.

        Returns:
            List of checkpoint paths.
        """
        if not os.path.exists(self._checkpoint_dir):
            return []

        checkpoints = []
        for filename in os.listdir(self._checkpoint_dir):
            if filename.startswith("rollout_ep") and filename.endswith(".pkl"):
                if episode is not None:
                    if not filename.startswith(f"rollout_ep{episode}_"):
                        continue

                checkpoint_path = os.path.join(self._checkpoint_dir, filename)
                checkpoints.append(checkpoint_path)

        # Sort by episode and batch_id
        def get_episode_batch(path):
            filename = os.path.basename(path)
            parts = (
                filename.replace("rollout_ep", "")
                .replace(".pkl", "")
                .split("_batch")
            )
            return (int(parts[0]), int(parts[1]))

        checkpoints.sort(key=get_episode_batch)
        return checkpoints

    def clear_episode_checkpoints(self, episode: int):
        """
        Remove all checkpoints for a specific episode.

        Args:
            episode: Episode number to clear.
        """
        checkpoints = self.list_rollout_batches(episode)

        for checkpoint_path in checkpoints:
            try:
                os.remove(checkpoint_path)
                logger.info(f"Removed rollout checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint: {e}")

    def clear_old_episodes(self, keep_last_n_episodes: int = 2):
        """
        Remove checkpoints from old episodes, keeping only recent ones.

        Args:
            keep_last_n_episodes: Number of recent episodes to keep.
        """
        checkpoints = self.list_rollout_batches()

        if not checkpoints:
            return

        # Extract unique episodes
        episodes = set()
        for checkpoint_path in checkpoints:
            filename = os.path.basename(checkpoint_path)
            episode = int(
                filename.replace("rollout_ep", "").split("_batch")[0]
            )
            episodes.add(episode)

        # Sort episodes
        sorted_episodes = sorted(episodes)

        # Remove old episodes
        if len(sorted_episodes) > keep_last_n_episodes:
            episodes_to_remove = sorted_episodes[:-keep_last_n_episodes]
            for episode in episodes_to_remove:
                self.clear_episode_checkpoints(episode)

