"""Tests for the CSI simulator module."""
import numpy as np
import pytest
import tempfile
import os

from csi_node.simulator import (
    CSISimulator, SimScenario, generate_sample_log,
    ENVIRONMENT_PRESETS,
)


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

    def test_labels_include_v2_fields(self):
        sim = CSISimulator(seed=42)
        items = list(sim.stream_with_labels(loop=False))
        _, label = items[0]
        assert "n_people" in label
        assert "distance_m" in label


class TestCSISimulatorPhysics:
    """Tests for physics-based simulation mode."""

    def test_physics_mode_basic(self):
        sim = CSISimulator(seed=42, use_physics=True)
        pkts = list(sim.stream(loop=False, realtime=False))
        assert len(pkts) > 100

    def test_physics_packet_format(self):
        sim = CSISimulator(seed=42, use_physics=True, n_subcarriers=52)
        pkt = next(sim.stream(loop=False, realtime=False))
        assert "ts" in pkt
        assert "csi" in pkt
        assert "rssi" in pkt
        assert pkt["csi"].shape == (52,)
        assert np.all(pkt["csi"] > 0)

    def test_physics_amplitudes_positive(self):
        sim = CSISimulator(seed=42, use_physics=True)
        for pkt in sim.stream(loop=False, realtime=False):
            assert np.all(pkt["csi"] > 0)

    def test_wall_materials_param(self):
        sim = CSISimulator(seed=42, wall_materials=["drywall", "concrete_thin"])
        assert sim.use_physics is True
        info = sim.channel_info
        assert info["mode"] == "physics"
        assert "Drywall" in info["walls"][0]
        assert "Concrete" in info["walls"][1]

    def test_environment_preset(self):
        for env_name in ENVIRONMENT_PRESETS:
            sim = CSISimulator(seed=42, environment=env_name)
            assert sim.use_physics is True
            info = sim.channel_info
            assert info["mode"] == "physics"

    def test_unknown_wall_material_raises(self):
        with pytest.raises(ValueError, match="Unknown wall material"):
            CSISimulator(seed=42, wall_materials=["unobtanium"])

    def test_concrete_higher_wall_loss_than_drywall(self):
        """Concrete walls should have higher wall loss than drywall."""
        sim_drywall = CSISimulator(seed=42, wall_materials=["drywall"])
        sim_concrete = CSISimulator(seed=42, wall_materials=["concrete_thick"])
        # Verify the channel model reports higher loss for concrete
        drywall_loss = sim_drywall._channel.total_wall_loss_db
        concrete_loss = sim_concrete._channel.total_wall_loss_db
        assert concrete_loss > drywall_loss * 2  # Concrete should be much worse

    def test_physics_presence_detection(self):
        """Physics mode should still show presence vs empty difference."""
        sim = CSISimulator(seed=42, use_physics=True)
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
        # Presence should increase energy (body reflection)
        assert np.mean(present_energies) > np.mean(empty_energies) * 1.05

    def test_no_interference_option(self):
        sim = CSISimulator(seed=42, use_physics=True, enable_interference=False)
        info = sim.channel_info
        assert info["interference"] is False

    def test_channel_info_legacy(self):
        sim = CSISimulator(seed=42)
        assert sim.channel_info == {"mode": "legacy"}

    def test_channel_info_physics(self):
        sim = CSISimulator(seed=42, use_physics=True)
        info = sim.channel_info
        assert info["mode"] == "physics"
        assert "total_wall_loss_db" in info
        assert "free_space_loss_db" in info
        assert "distance_m" in info

    def test_through_wall_physics(self):
        sim = CSISimulator(seed=42, through_wall=True, use_physics=True,
                           wall_materials=["cinder_block"])
        pkts = list(sim.stream(loop=False, realtime=False))
        assert len(pkts) > 200


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
