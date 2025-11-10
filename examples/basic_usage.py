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

"""Basic usage example for chen1110 fault tolerance system."""

import logging
import time

import torch
import torch.nn as nn

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def example_resource_monitoring():
    """Example: Using resource monitoring."""
    from chen1110.agent.monitor.resource import ResourceMonitor

    logger.info("=== Resource Monitoring Example ===")

    # Create and start resource monitor
    monitor = ResourceMonitor.singleton_instance()

    # Manually report resource usage
    monitor.report_resource()

    # In production, you would start the monitoring thread
    # monitor.start()


def example_training_monitoring():
    """Example: Using training monitoring."""
    from chen1110.agent.monitor.training import TrainingMonitor, write_training_metrics

    logger.info("\n=== Training Monitoring Example ===")

    # Create training monitor
    metrics_path = "/tmp/training_metrics.json"
    monitor = TrainingMonitor.singleton_instance(metrics_path)

    # Write training metrics
    for step in range(5):
        write_training_metrics(metrics_path, step=step * 100)
        logger.info(f"Written metrics for step {step * 100}")
        time.sleep(0.5)


def example_diagnosis_agent():
    """Example: Using diagnosis agent."""
    from chen1110.controller.diagnosis import DiagnosisAgent

    logger.info("\n=== Diagnosis Agent Example ===")

    # Create diagnosis agent
    agent = DiagnosisAgent.singleton_instance(
        training_log_file="/tmp/training.log",
        node_rank=0,
        local_world_size=4,
    )

    # Start the agent (this starts all periodic collectors)
    agent.start()

    logger.info("Diagnosis agent started, collecting data...")
    time.sleep(3)

    # Diagnose a simulated failure
    action = agent.diagnose_training_failure(failures={}, restart_count=0)
    logger.info(f"Diagnosis action: {action.action_type.name} - {action.reason}")

    # Stop the agent
    agent.stop()
    logger.info("Diagnosis agent stopped")


def example_checkpoint_management():
    """Example: Using checkpoint managers."""
    from chen1110.ckpt_manager import (
        LatestCheckpointManager,
        PeriodicCheckpointManager,
        RefLogPCheckpointManager,
        RolloutResponseCheckpointManager,
    )

    logger.info("\n=== Checkpoint Management Example ===")

    # Create a simple model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())

    # 1. Latest Checkpoint Manager
    logger.info("\n1. Latest Checkpoint Manager:")
    latest_ckpt = LatestCheckpointManager("/tmp/checkpoints/latest", max_checkpoints=3)

    for step in [100, 200, 300]:
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        path = latest_ckpt.save(state_dict, step)
        logger.info(f"Saved checkpoint at step {step}: {path}")

    # Load latest
    loaded = latest_ckpt.load()
    if loaded:
        logger.info(f"Loaded checkpoint from step {loaded['step']}")

    # 2. Periodic Checkpoint Manager
    logger.info("\n2. Periodic Checkpoint Manager:")
    periodic_ckpt = PeriodicCheckpointManager(
        "/tmp/checkpoints/periodic", save_interval=500, max_checkpoints=5
    )

    for step in range(0, 2000, 250):
        state_dict = {"model": model.state_dict(), "step": step}
        saved = periodic_ckpt.save_if_needed(state_dict, step)
        if saved:
            logger.info(f"Periodic checkpoint saved at step {step}")
        else:
            logger.info(f"Skipped checkpoint at step {step}")

    # 3. Reference LogP Checkpoint Manager (RL-specific)
    logger.info("\n3. Reference LogP Checkpoint Manager:")
    ref_logp_ckpt = RefLogPCheckpointManager("/tmp/checkpoints/ref_logp")

    logp_data = torch.randn(10, 50)  # Example log probabilities
    ref_logp_ckpt.save_ref_logp(logp_data, episode=1, step=100)
    logger.info("Saved reference LogP checkpoint")

    loaded_logp = ref_logp_ckpt.load_ref_logp(episode=1, step=100)
    if loaded_logp is not None:
        logger.info(f"Loaded reference LogP with shape {loaded_logp.shape}")

    # 4. Rollout Response Checkpoint Manager (RL-specific)
    logger.info("\n4. Rollout Response Checkpoint Manager:")
    rollout_ckpt = RolloutResponseCheckpointManager("/tmp/checkpoints/rollout")

    responses = [
        {
            "prompt": f"Question {i}",
            "response": f"Answer {i}",
            "reward": float(i * 0.5),
        }
        for i in range(5)
    ]

    rollout_ckpt.save_rollout_batch(responses, episode=1, batch_id=0)
    logger.info(f"Saved {len(responses)} rollout responses")

    loaded_responses = rollout_ckpt.load_rollout_batch(episode=1, batch_id=0)
    if loaded_responses:
        logger.info(
            f"Loaded {loaded_responses['num_samples']} rollout responses from episode {loaded_responses['episode']}"
        )


def main():
    """Run all examples."""
    logger.info("=== chen1110 RL Fault Tolerance System Examples ===\n")

    try:
        # Run examples
        example_resource_monitoring()
        example_training_monitoring()
        example_checkpoint_management()
        example_diagnosis_agent()

        logger.info("\n=== All examples completed successfully! ===")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()

