"""Tests for the CSI simulator module."""
import numpy as np
import pytest
import tempfile
import os

from csi_node.simulator import CSISimulator, SimScenario, generate_sample_log


class TestCSISimulator:
    def test_basic_stream(self):
        sim = CSISimulator(seed=42)
        pkts = list(sim.stream(loop=False, realtime=False))
        assert len(pkts) > 100  # Should produce many packets

    def test_packet_format(self):
        sim = CSISimulator(n_subcarriers=52, seed=42)
        pkts = list(sim.stream(loop=False, realtime=False))
        pkt = pkts[0]
        assert "ts" in pkt
        assert "csi" in pkt
        assert "rssi" in pkt
        assert isinstance(pkt["csi"], np.ndarray)
        assert pkt["csi"].shape == (52,)
        assert len(pkt["rssi"]) == 2

    def test_amplitudes_positive(self):
        sim = CSISimulator(seed=42)
        for pkt in sim.stream(loop=False, realtime=False):
            assert np.all(pkt["csi"] > 0)

    def test_presence_changes_energy(self):
        """Presence scenarios should have higher energy than empty room."""
        sim = CSISimulator(seed=42)
        empty_energies = []
        present_energies = []
        for pkt, label in sim.stream_with_labels(loop=False):
            energy = float(np.sum(pkt["csi"] ** 2))
            if label["presence"]:
                present_energies.append(energy)
            else:
                empty_energies.append(energy)
        
        assert len(empty_energies) > 0
        assert len(present_energies) > 0
        assert np.mean(present_energies) > np.mean(empty_energies) * 1.5

    def test_custom_scenarios(self):
        scenarios = [
            SimScenario("test_empty", 1.0, False, "none"),
            SimScenario("test_present", 1.0, True, "stationary", 2.0, 3.0),
        ]
        sim = CSISimulator(scenarios=scenarios, seed=42)
        pkts = list(sim.stream(loop=False, realtime=False))
        assert len(pkts) == int(2.0 * 30)  # 2 seconds at 30Hz

    def test_stream_with_labels(self):
        sim = CSISimulator(seed=42)
        items = list(sim.stream_with_labels(loop=False))
        assert len(items) > 0
        pkt, label = items[0]
        assert "scenario" in label
        assert "presence" in label
        assert "movement" in label

    def test_different_seeds_different_data(self):
        sim1 = CSISimulator(seed=1)
        sim2 = CSISimulator(seed=2)
        pkt1 = next(sim1.stream(loop=False, realtime=False))
        pkt2 = next(sim2.stream(loop=False, realtime=False))
        assert not np.array_equal(pkt1["csi"], pkt2["csi"])


class TestGenerateSampleLog:
    def test_generates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            generate_sample_log(path, duration_s=2.0, seed=42)
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) > 30  # At least 1 second of data at 30Hz
        finally:
            os.unlink(path)
