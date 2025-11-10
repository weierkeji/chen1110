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

"""Basic tests for chen1110 components."""

import os
import tempfile

import pytest
import torch


class TestDataCollectors:
    """Test data collectors."""

    def test_resource_collector(self):
        """Test resource collector."""
        from chen1110.agent.data_collector import ResourceCollector

        collector = ResourceCollector()
        assert collector.is_enabled()

        # Collect data (should not raise)
        collector.collect_data()

    def test_metric_collector(self):
        """Test metric collector."""
        from chen1110.agent.data_collector import MetricCollector

        collector = MetricCollector()
        # Without XPUTimer, it should be disabled
        # This is expected behavior
        data = collector.collect_data()
        assert data == ""

    def test_log_collector(self):
        """Test log collector."""
        from chen1110.agent.data_collector import LogCollector

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("Test log line 1\n")
            f.write("Test log line 2\n")
            log_path = f.name

        try:
            collector = LogCollector(log_path)
            assert collector.is_enabled()

            data = collector.collect_data()
            assert "Test log line" in data

        finally:
            os.unlink(log_path)

    def test_stack_collector(self):
        """Test stack collector."""
        from chen1110.agent.data_collector import StackCollector

        collector = StackCollector()
        assert collector.is_enabled()

        stacks = collector.collect_data()
        assert isinstance(stacks, dict)
        assert len(stacks) > 0  # Should have at least main thread


class TestCheckpointManagers:
    """Test checkpoint managers."""

    def test_latest_checkpoint_manager(self):
        """Test latest checkpoint manager."""
        from chen1110.ckpt_manager import LatestCheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = LatestCheckpointManager(tmpdir, max_checkpoints=2)

            # Save checkpoints
            for step in [100, 200, 300]:
                state_dict = {"step": step, "data": torch.randn(10, 10)}
                path = manager.save(state_dict, step)
                assert os.path.exists(path)

            # Should have max 2 checkpoints
            checkpoints = manager._list_checkpoints()
            assert len(checkpoints) <= 2

            # Load latest
            loaded = manager.load()
            assert loaded is not None
            assert loaded["step"] == 300

    def test_periodic_checkpoint_manager(self):
        """Test periodic checkpoint manager."""
        from chen1110.ckpt_manager import PeriodicCheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PeriodicCheckpointManager(
                tmpdir, save_interval=500, max_checkpoints=3
            )

            # Test save_if_needed
            for step in range(0, 1500, 100):
                state_dict = {"step": step}
                saved = manager.save_if_needed(state_dict, step)

                # Should save at 0, 500, 1000
                if step in [500, 1000]:
                    assert saved
                elif step == 0:
                    assert not saved  # Don't save at step 0

    def test_ref_logp_checkpoint_manager(self):
        """Test reference LogP checkpoint manager."""
        from chen1110.ckpt_manager import RefLogPCheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RefLogPCheckpointManager(tmpdir)

            # Save reference LogP
            logp_data = torch.randn(10, 50)
            path = manager.save_ref_logp(logp_data, episode=1, step=100)
            assert os.path.exists(path)

            # Load reference LogP
            loaded_logp = manager.load_ref_logp(episode=1, step=100)
            assert loaded_logp is not None
            assert loaded_logp.shape == logp_data.shape


class TestUtils:
    """Test utility functions."""

    def test_singleton(self):
        """Test singleton pattern."""
        from chen1110.common.utils import Singleton

        class TestClass(Singleton):
            def __init__(self, value):
                self.value = value

        instance1 = TestClass.singleton_instance(value=42)
        instance2 = TestClass.singleton_instance(value=99)

        # Should be the same instance
        assert instance1 is instance2
        assert instance1.value == 42  # First value is kept


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

