"""
tests/test_sgi.py
─────────────────
Full test suite for the sgi package.
Run with: pytest tests/ -v

Author: Yahya Akbay | 2025
"""

import numpy as np
import warnings
import pytest

import sgi
from sgi import (
    SGILightClassifier, SGIFeatureExtractor,
    generate_window, generate_dataset,
    k_amplitude, k_to_delta_g, collective_k, gap_to_detector,
    EPSILON, DEFAULT_CLASSES, FEATURE_NAMES,
)


# ── Physics tests ─────────────────────────────────────────────────────────────

class TestPhysics:
    def test_epsilon_value(self):
        """Earth surface relativistic correction ~1.39e-9"""
        assert 1.3e-9 < EPSILON < 1.5e-9

    def test_k_amplitude_human(self):
        """K(human, 1.4 m/s) should be ~1.8e-52 m⁻¹"""
        K = k_amplitude(80, 1.4)
        assert 1e-53 < K < 1e-51

    def test_k_amplitude_scales_with_mass(self):
        """Heavier objects → larger K"""
        K_human = k_amplitude(80, 10)
        K_truck = k_amplitude(15000, 10)
        assert K_truck > K_human * 100

    def test_k_amplitude_scales_with_v2(self):
        """K scales as v²"""
        K1 = k_amplitude(80, 1.0)
        K2 = k_amplitude(80, 2.0)
        ratio = K2 / K1
        assert 3.9 < ratio < 4.1  # should be ~4

    def test_collective_coherent(self):
        """Coherent sum: N × K"""
        K1 = k_amplitude(80, 1.4)
        Kc = collective_k(K1, 100, mode='coherent')
        assert abs(Kc / K1 - 100) < 1

    def test_collective_incoherent(self):
        """Incoherent sum: √N × K"""
        K1  = k_amplitude(80, 1.4)
        Kinc = collective_k(K1, 100, mode='incoherent')
        assert abs(Kinc / K1 - 10) < 0.5

    def test_collective_invalid_mode(self):
        with pytest.raises(ValueError):
            collective_k(1e-50, 10, mode='quantum_magic')

    def test_gap_to_detector(self):
        """Gap for single human vs quantum gravimeter: ~27 orders"""
        result = gap_to_detector(80, 1.4,
                                  detector_sensitivity_ms2=1e-9)
        assert result['gap_orders'] > 20
        assert result['n_coherent'] > 1e20

    def test_k_to_dg(self):
        """δg = K · c² / r, should be tiny"""
        K  = k_amplitude(80, 1.4)
        dg = k_to_delta_g(K)
        assert dg < 1e-30


# ── Feature extractor tests ───────────────────────────────────────────────────

class TestFeatureExtractor:
    def setup_method(self):
        self.ext = SGIFeatureExtractor(fs=100.0)

    def test_output_shape(self):
        w    = generate_window('human', seed=42)
        feat = self.ext.extract(w)
        assert feat.shape == (len(FEATURE_NAMES),)
        assert len(FEATURE_NAMES) == 14

    def test_output_dtype(self):
        w    = generate_window('car', seed=1)
        feat = self.ext.extract(w)
        assert feat.dtype == np.float32

    def test_no_nan(self):
        for cls in DEFAULT_CLASSES:
            w    = generate_window(cls, seed=0)
            feat = self.ext.extract(w)
            assert not np.any(np.isnan(feat)), f"NaN in {cls}"

    def test_no_vib_bandwidth(self):
        """vib_bandwidth must have been removed (dead feature, 0% importance)"""
        assert 'vib_bandwidth' not in FEATURE_NAMES

    def test_gps_velocity_warning_constant(self):
        """Near-constant velocity should trigger a UserWarning"""
        w = generate_window('human', seed=42)
        # Override velocity with near-zero constant signal
        n = len(w['velocity'])
        w['velocity'] = np.zeros(n)
        with pytest.warns(UserWarning, match="constant or near-zero"):
            self.ext.extract(w)

    def test_gps_velocity_no_warning_real(self):
        """Real GPS velocity (non-constant) must NOT trigger a warning"""
        w = generate_window('car', seed=42)
        # car windows have realistic non-zero varying velocity
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            self.ext.extract(w)  # should not raise

    def test_vib_freq_drone(self):
        """Drone rotor should produce peak ~50 Hz"""
        w    = generate_window('drone', seed=42, fs=200)
        ext  = SGIFeatureExtractor(fs=200)
        feat = ext.extract(w)
        vib_freq_idx = FEATURE_NAMES.index('vib_freq')
        assert feat[vib_freq_idx] > 30, "Drone vib_freq should be >30 Hz"

    def test_vib_freq_human(self):
        """Human gait should produce peak ~1-3 Hz"""
        w    = generate_window('human', seed=42)
        feat = self.ext.extract(w)
        vib_freq_idx = FEATURE_NAMES.index('vib_freq')
        assert feat[vib_freq_idx] < 10, "Human vib_freq should be <10 Hz"

    def test_vib_freq_ratio_car_lt_truck(self):
        """car vib_freq_ratio must be lower than truck — core physical discriminator"""
        ext = SGIFeatureExtractor(fs=100.0)
        ratio_idx = FEATURE_NAMES.index('vib_freq_ratio')
        # Average over multiple seeds for stability
        car_ratios   = [ext.extract(generate_window('car',   seed=i))[ratio_idx] for i in range(20)]
        truck_ratios = [ext.extract(generate_window('truck', seed=i))[ratio_idx] for i in range(20)]
        assert np.mean(car_ratios) < np.mean(truck_ratios), (
            f"car vib_freq_ratio mean {np.mean(car_ratios):.4f} should be "
            f"< truck {np.mean(truck_ratios):.4f}"
        )

    def test_vib_freq_ratio_range(self):
        """vib_freq_ratio must be in [0, 1] for all classes"""
        ext = SGIFeatureExtractor(fs=100.0)
        ratio_idx = FEATURE_NAMES.index('vib_freq_ratio')
        for cls in DEFAULT_CLASSES:
            w = generate_window(cls, seed=42)
            ratio = ext.extract(w)[ratio_idx]
            assert 0.0 <= ratio <= 1.0, \
                f"{cls} vib_freq_ratio={ratio:.4f} out of [0,1]"

    def test_heading_norm_car_gt_truck(self):
        """car heading_norm must exceed truck — car makes sharper turns relative to speed"""
        ext = SGIFeatureExtractor(fs=100.0)
        norm_idx = FEATURE_NAMES.index('heading_norm')
        car_norms   = [ext.extract(generate_window('car',   seed=i))[norm_idx] for i in range(20)]
        truck_norms = [ext.extract(generate_window('truck', seed=i))[norm_idx] for i in range(20)]
        assert np.mean(car_norms) > np.mean(truck_norms), (
            f"car heading_norm mean {np.mean(car_norms):.4f} should be "
            f"> truck {np.mean(truck_norms):.4f}"
        )

    def test_heading_norm_positive(self):
        """heading_norm and omega_norm must be non-negative for all classes"""
        ext = SGIFeatureExtractor(fs=100.0)
        heading_norm_idx = FEATURE_NAMES.index('heading_norm')
        omega_norm_idx   = FEATURE_NAMES.index('omega_norm')
        for cls in DEFAULT_CLASSES:
            w = generate_window(cls, seed=42)
            feat = ext.extract(w)
            assert feat[heading_norm_idx] >= 0.0, f"{cls} heading_norm negative"
            assert feat[omega_norm_idx]   >= 0.0, f"{cls} omega_norm negative"

    def test_array_input(self):
        """Test numpy array input (N,6)"""
        N   = 500
        arr = np.random.randn(N, 6).astype(np.float32)
        arr[:, 0] = np.abs(arr[:, 0]) + 1.0  # positive velocity
        feat = self.ext.extract_array(arr)
        assert feat.shape == (len(FEATURE_NAMES),)

    def test_batch(self):
        """Batch extraction returns (N_windows, N_features)"""
        windows = [generate_window(c, seed=i) for i, c in enumerate(DEFAULT_CLASSES)]
        feats   = self.ext.extract_batch(windows)
        assert feats.shape == (len(DEFAULT_CLASSES), len(FEATURE_NAMES))


# ── Generator tests ───────────────────────────────────────────────────────────

class TestGenerator:
    def test_window_keys(self):
        w = generate_window('human')
        required = {'velocity','a_long','a_lat','a_vert','omega_z','heading_rate'}
        assert required.issubset(set(w.keys()))

    def test_window_length(self):
        w = generate_window('car', duration=5.0, fs=100.0)
        assert len(w['velocity']) == 500

    def test_reproducibility(self):
        w1 = generate_window('truck', seed=42)
        w2 = generate_window('truck', seed=42)
        np.testing.assert_array_equal(w1['velocity'], w2['velocity'])

    def test_unknown_class(self):
        with pytest.raises(ValueError):
            generate_window('spaceship')

    def test_dataset_shape(self):
        X, y = generate_dataset(n_per_class=10)
        assert X.shape == (10 * len(DEFAULT_CLASSES), len(FEATURE_NAMES))
        assert len(y) == len(X)

    def test_dataset_labels(self):
        X, y = generate_dataset(n_per_class=5)
        for cls in DEFAULT_CLASSES:
            assert cls in y

    def test_car_vib_amp_realistic(self):
        """car a_vert std must be non-zero (road noise modelled)"""
        w = generate_window('car', seed=42)
        assert np.std(w['a_vert']) > 0.001, \
            f"car a_vert std too low: {np.std(w['a_vert']):.4f} — road noise not modelled"

    def test_truck_speed_lt_car(self):
        """truck v_mean must be less than car — primary urban speed discriminator"""
        from sgi._internal.generator import MOTION_PARAMS
        assert MOTION_PARAMS['truck'][0] < MOTION_PARAMS['car'][0], \
            "truck v_mean should be less than car (urban: truck ~30km/h, car ~50km/h)"

    def test_truck_vib_amp_gt_car(self):
        """truck a_vert std must exceed car (heavier vehicle, rougher vibration)"""
        car_w   = generate_window('car',   seed=42)
        truck_w = generate_window('truck', seed=42)
        assert np.std(truck_w['a_vert']) > np.std(car_w['a_vert']), \
            "truck a_vert std should exceed car"

    def test_truck_a_long_gt_car(self):
        """truck a_long must exceed car — heavier vehicle brakes harder"""
        from sgi._internal.generator import MOTION_PARAMS
        assert MOTION_PARAMS['truck'][2] > MOTION_PARAMS['car'][2], \
            "truck a_long should exceed car (heavier vehicle, longer braking)"

    def test_car_v_std_gt_truck(self):
        """car v_std must exceed truck — city stop-and-go vs steady truck cruise"""
        from sgi._internal.generator import MOTION_PARAMS
        assert MOTION_PARAMS['car'][1] > MOTION_PARAMS['truck'][1], \
            "car v_std should exceed truck (city stop-and-go vs steady cruise)"


# ── Classifier tests ──────────────────────────────────────────────────────────

class TestClassifier:
    @pytest.fixture(scope='class')
    def trained_clf(self):
        clf = SGILightClassifier()
        clf.train(n_per_class=50, verbose=False)
        return clf

    def test_predict_returns_string(self, trained_clf):
        w      = generate_window('car', seed=1)
        result = trained_clf.predict(w)
        assert isinstance(result, str)
        assert result in DEFAULT_CLASSES

    def test_predict_proba_sums_to_one(self, trained_clf):
        w     = generate_window('truck', seed=2)
        proba = trained_clf.predict_proba(w)
        assert abs(sum(proba.values()) - 1.0) < 1e-5

    def test_predict_proba_keys(self, trained_clf):
        w     = generate_window('human', seed=3)
        proba = trained_clf.predict_proba(w)
        assert set(proba.keys()) == set(DEFAULT_CLASSES)

    def test_array_input(self, trained_clf):
        """Classifier accepts numpy (N,6) array"""
        arr = np.random.randn(500, 6).astype(np.float32)
        arr[:, 0] = np.abs(arr[:, 0]) + 1.0
        result = trained_clf.predict(arr)
        assert result in DEFAULT_CLASSES

    def test_untrained_raises(self):
        clf = SGILightClassifier()
        w   = generate_window('human')
        with pytest.raises(RuntimeError):
            clf.predict(w)

    def test_accuracy_reasonable(self, trained_clf):
        """Should achieve >70% on held-out data (conservative threshold)"""
        X, y    = generate_dataset(n_per_class=30, seed_offset=88888)
        y_pred  = trained_clf.pipeline.predict(X)
        acc     = np.mean(y_pred == y)
        assert acc > 0.70, f"Accuracy {acc:.2%} below threshold"

    def test_save_load(self, trained_clf, tmp_path):
        """Model survives save/load cycle"""
        path = str(tmp_path / "test_model.pkl")
        trained_clf.save(path)
        loaded = SGILightClassifier.load(path)
        w      = generate_window('drone', seed=99)
        assert trained_clf.predict(w) == loaded.predict(w)

    def test_evaluate_returns_array(self, trained_clf):
        scores = trained_clf.evaluate(n_per_class=20, cv=3, verbose=False)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert trained_clf.cv_scores_ is not None

    def test_summary_runs(self, trained_clf, capsys):
        trained_clf.summary()
        out = capsys.readouterr().out
        assert 'SGI-Light' in out

    def test_predict_batch(self, trained_clf):
        windows = [generate_window(c, seed=i+100) for i, c in enumerate(DEFAULT_CLASSES)]
        results = trained_clf.predict_batch(windows)
        assert isinstance(results, list)
        assert len(results) == len(DEFAULT_CLASSES)
        assert all(r in DEFAULT_CLASSES for r in results)


# ── Top-level API tests ───────────────────────────────────────────────────────

class TestTopLevelAPI:
    def test_train_returns_classifier(self):
        clf = sgi.train(n_per_class=30, verbose=False)
        assert isinstance(clf, SGILightClassifier)
        assert clf.is_trained

    def test_predict_convenience(self):
        w = generate_window('bicycle', seed=7)
        r = sgi.predict(w)
        assert r in DEFAULT_CLASSES

    def test_predict_proba_convenience(self):
        w = generate_window('car', seed=8)
        p = sgi.predict_proba(w)
        assert isinstance(p, dict)
        assert abs(sum(p.values()) - 1.0) < 1e-5

    def test_info_runs(self, capsys):
        sgi.info()
        out = capsys.readouterr().out
        assert 'sgi' in out.lower()
        assert 'epsilon' in out.lower() or 'ε' in out.lower()

    def test_version(self):
        assert sgi.__version__ == '1.0.0'
