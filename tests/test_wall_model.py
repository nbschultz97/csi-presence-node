"""Tests for wall attenuation and channel models."""
import numpy as np
import pytest
import math

from csi_node.wall_model import (
    WallMaterial,
    MATERIALS,
    ChannelModel,
    HumanBodyModel,
    InterferenceModel,
    MultipathComponent,
)


class TestWallMaterial:
    def test_known_materials_exist(self):
        expected = ["drywall", "concrete_thin", "concrete_thick", "brick",
                    "wood", "glass", "metal", "cinder_block", "none"]
        for name in expected:
            assert name in MATERIALS

    def test_attenuation_at_base_freq(self):
        drywall = MATERIALS["drywall"]
        assert drywall.attenuation_db(2.4) == pytest.approx(drywall.base_loss_db)

    def test_attenuation_increases_with_freq(self):
        concrete = MATERIALS["concrete_thin"]
        loss_24 = concrete.attenuation_db(2.4)
        loss_50 = concrete.attenuation_db(5.0)
        assert loss_50 > loss_24

    def test_no_wall_zero_loss(self):
        none = MATERIALS["none"]
        assert none.attenuation_db(2.4) == 0.0
        assert none.attenuation_db(5.0) == 0.0
        assert none.phase_shift_rad == 0.0

    def test_metal_highest_loss(self):
        metal_loss = MATERIALS["metal"].attenuation_db(2.4)
        for name, mat in MATERIALS.items():
            if name != "metal":
                assert metal_loss >= mat.attenuation_db(2.4)

    def test_drywall_double_more_than_single(self):
        single = MATERIALS["drywall"].attenuation_db(2.4)
        double = MATERIALS["drywall_double"].attenuation_db(2.4)
        assert double > single


class TestChannelModel:
    def test_basic_creation(self):
        ch = ChannelModel(rng=np.random.default_rng(42))
        assert ch.n_subcarriers == 52
        assert ch.freq_ghz == 2.4

    def test_channel_response_shape(self):
        ch = ChannelModel(n_subcarriers=64, rng=np.random.default_rng(42))
        h = ch.channel_response(0.0)
        assert h.shape == (64,)
        assert h.dtype == complex

    def test_amplitude_response_positive(self):
        ch = ChannelModel(rng=np.random.default_rng(42))
        amps = ch.amplitude_response(0.0)
        assert np.all(amps >= 0)
        assert amps.shape == (52,)

    def test_wall_loss_additive(self):
        walls_1 = [MATERIALS["drywall"]]
        walls_2 = [MATERIALS["drywall"], MATERIALS["drywall"]]
        ch1 = ChannelModel(walls=walls_1, rng=np.random.default_rng(42))
        ch2 = ChannelModel(walls=walls_2, rng=np.random.default_rng(42))
        assert ch2.total_wall_loss_db == pytest.approx(2 * ch1.total_wall_loss_db)

    def test_free_space_loss_increases_with_distance(self):
        ch_near = ChannelModel(distance_m=2.0, rng=np.random.default_rng(42))
        ch_far = ChannelModel(distance_m=10.0, rng=np.random.default_rng(42))
        assert ch_far.free_space_loss_db() > ch_near.free_space_loss_db()

    def test_free_space_loss_zero_distance(self):
        ch = ChannelModel(distance_m=0.0, rng=np.random.default_rng(42))
        assert ch.free_space_loss_db() == 0.0

    def test_time_varying_response(self):
        ch = ChannelModel(rng=np.random.default_rng(42))
        h0 = ch.amplitude_response(0.0)
        h1 = ch.amplitude_response(1.0)
        # Should differ due to Doppler evolution
        assert not np.array_equal(h0, h1)

    def test_concrete_attenuates_more_than_drywall(self):
        ch_drywall = ChannelModel(
            walls=[MATERIALS["drywall"]], rng=np.random.default_rng(42))
        ch_concrete = ChannelModel(
            walls=[MATERIALS["concrete_thin"]], rng=np.random.default_rng(42))
        assert ch_concrete.total_wall_loss_db > ch_drywall.total_wall_loss_db

    def test_rician_k_affects_response(self):
        ch_high_k = ChannelModel(rician_k=10.0, rng=np.random.default_rng(42))
        ch_low_k = ChannelModel(rician_k=0.0, rng=np.random.default_rng(42))
        # High K = stronger LOS = less variance across subcarriers
        h_high = ch_high_k.amplitude_response(0.0)
        h_low = ch_low_k.amplitude_response(0.0)
        assert np.std(h_high) / np.mean(h_high) <= np.std(h_low) / np.mean(h_low) + 0.5


class TestHumanBodyModel:
    def test_default_creation(self):
        body = HumanBodyModel()
        assert body.breath_rate_hz > 0
        assert body.reflection_coeff > 0

    def test_body_doppler(self):
        body = HumanBodyModel()
        d = body.body_doppler(1.0, 2.4)
        # Walking at 1 m/s at 2.4GHz: wavelength=0.125m, doppler=2*1/0.125=16Hz
        assert d == pytest.approx(16.0, rel=0.01)

    def test_zero_velocity_zero_doppler(self):
        body = HumanBodyModel()
        assert body.body_doppler(0.0) == 0.0

    def test_breathing_modulation_shape(self):
        body = HumanBodyModel()
        mod = body.breathing_modulation(0.0, 52)
        assert mod.shape == (52,)

    def test_breathing_modulation_periodic(self):
        body = HumanBodyModel()
        period = 1.0 / body.breath_rate_hz
        mod0 = body.breathing_modulation(0.0, 52)
        mod_period = body.breathing_modulation(period, 52)
        # Should be approximately equal after one full period
        np.testing.assert_allclose(mod0, mod_period, atol=0.05)

    def test_breathing_central_subcarriers_stronger(self):
        body = HumanBodyModel()
        # At peak breathing (t where sin is ~1)
        t = 1.0 / (4 * body.breath_rate_hz)
        mod = body.breathing_modulation(t, 52)
        center_effect = abs(mod[26])
        edge_effect = abs(mod[0])
        assert center_effect >= edge_effect

    def test_walking_doppler_pattern_shape(self):
        body = HumanBodyModel()
        pattern = body.walking_doppler_pattern(0.0, 1.0, 52)
        assert pattern.shape == (52,)

    def test_walking_doppler_zero_speed(self):
        body = HumanBodyModel()
        pattern = body.walking_doppler_pattern(0.0, 0.0, 52)
        np.testing.assert_allclose(pattern, 0.0, atol=1e-10)


class TestInterferenceModel:
    def test_default_creation(self):
        intf = InterferenceModel()
        assert intf.burst_probability >= 0
        assert intf.burst_probability <= 1

    def test_interference_shape(self):
        intf = InterferenceModel()
        rng = np.random.default_rng(42)
        noise = intf.interference(0.0, 52, rng)
        assert noise.shape == (52,)

    def test_interference_not_all_zero(self):
        intf = InterferenceModel()
        rng = np.random.default_rng(42)
        total = np.zeros(52)
        for i in range(100):
            total += np.abs(intf.interference(i * 0.033, 52, rng))
        assert np.sum(total) > 0

    def test_burst_eventually_occurs(self):
        intf = InterferenceModel(burst_probability=0.5)
        rng = np.random.default_rng(42)
        burst_seen = False
        for i in range(200):
            noise = intf.interference(i * 0.033, 52, rng)
            if np.max(np.abs(noise)) > 1.0:
                burst_seen = True
                break
        assert burst_seen

    def test_no_burst_when_probability_zero(self):
        intf = InterferenceModel(burst_probability=0.0)
        rng = np.random.default_rng(42)
        for i in range(100):
            noise = intf.interference(i * 0.033, 52, rng)
            # Only background + beacon, no burst
            assert np.max(np.abs(noise)) < 20.0
