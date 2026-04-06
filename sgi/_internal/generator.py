"""
sgi._internal.generator
────────────────────────
Synthetic GPS+IMU motion profile generator.

Generates physically motivated training data for each object class.
Parameters derived from published biomechanics and vehicle dynamics literature.

Author: Yahya Akbay | 2025
"""

import numpy as np

# ── Motion parameters per class ───────────────────────────────────────────────
# Columns: (v_mean, v_std, a_long_std, a_lat_std, vib_freq, vib_amp, heading_std, road_noise_amp)
#
# road_noise_amp: class-specific road surface noise amplitude [deg/s]
#   - human, drone: 0.0 — no road surface contact (walking / airborne)
#   - bicycle:      0.5 — light road contact, thin tires, minimal IMU coupling
#   - car, truck:   3.0 — EQUAL road noise — same road surface, same dashboard coupling
#                         This is the key invariant: car and truck cannot be separated
#                         by heading_rms alone; classifier must use vib_freq and v_mean.
#
# Sources:
#   Human gait:      Menz et al. (2003)     → ~1.8 Hz
#   Bicycle cadence: Wilson (2004)           → ~2.5 Hz
#   Car engine idle: 4-cyl, 600 RPM         → ~5 Hz
#   Truck diesel:    6-cyl, 800 RPM         → ~8 Hz
#   Drone rotor:     DJI Mini series specs   → ~50 Hz
#   Calibrated against PVS dataset (Menegazzo, 2020) — Brazil, dashboard IMU
#   car v_mean = 9.5 m/s (34.2 km/h) — direct measurement from PVS GPS speed
#   (PVS mean GPS speed: 34.5 km/h, n=1500 windows)
MOTION_PARAMS = {
    #              v_mean  v_std  a_long  a_lat  vib_freq  vib_amp  heading_std  road_noise
    'human':   (   1.4,   0.30,  0.05,  0.03,    1.8,    0.08,     15.0,        0.0),
    'bicycle': (   4.5,   0.80,  0.10,  0.05,    2.5,    0.04,      8.0,        0.5),
    'car':     (   9.5,   3.50,  0.80,  0.40,    5.0,    0.02,      5.0,        3.0),  # v_mean calibrated to PVS: 34.5 km/h mean
    'truck':   (  11.1,   1.20,  1.40,  0.25,    8.0,    0.08,      1.5,        3.0),
    'drone':   (   8.0,   1.50,  0.60,  0.60,   50.0,    0.03,     25.0,        0.0),
}

DEFAULT_CLASSES = list(MOTION_PARAMS.keys())


def generate_window(obj_class: str,
                     duration: float = 5.0,
                     fs: float = 100.0,
                     seed: int = None) -> dict:
    """
    Generate one synthetic GPS+IMU window for a given object class.

    Parameters
    ----------
    obj_class : str   — one of MOTION_PARAMS keys
    duration  : float — window duration [s]
    fs        : float — IMU sample rate [Hz]
    seed      : int   — random seed for reproducibility

    Returns
    -------
    dict with keys:
        obj_class, t, velocity, a_long, a_lat, a_vert,
        omega_z, heading_rate, fs

    Note: Road Surface Noise
    ------------------------
    heading_rate and omega_z include a class-specific road surface noise component.
    human and drone receive zero road noise (not in contact with road surface).
    car and truck receive equal road noise (rn=3.0 deg/s) — same road surface,
    same dashboard IMU coupling. This is the key invariant that forces the
    classifier to rely on vib_freq, v_mean, and ke_proxy for car/truck separation
    rather than heading_rms, which carries equal noise for both vehicle classes.
    Calibrated against PVS dataset (Menegazzo, 2020), Brazil, dashboard IMU.
    """
    if obj_class not in MOTION_PARAMS:
        raise ValueError(f"Unknown class '{obj_class}'. "
                         f"Available: {list(MOTION_PARAMS.keys())}")

    rng = np.random.default_rng(seed)

    vm, vs, al_s, at_s, vf, va, hs, rn = MOTION_PARAMS[obj_class]
    N  = int(duration * fs)
    t  = np.arange(N) / fs
    ph = rng.uniform(0, 2 * np.pi)

    # Forward velocity
    v = np.clip(vm + vs * rng.standard_normal(N), 0.05, None)

    # Longitudinal acceleration
    a_long = (al_s * rng.standard_normal(N)
              + 0.3 * al_s * np.sin(2 * np.pi * 0.15 * t))

    # Lateral acceleration
    a_lat = at_s * rng.standard_normal(N)

    # Vertical vibration: dominant frequency + 2nd harmonic + broadband road noise
    # Calibrated against PVS dataset (Brazil, mixed road surfaces, 100 Hz)
    a_vert = (va * np.sin(2 * np.pi * vf * t + ph)          # dominant frequency
              + 0.5 * va * np.sin(2 * np.pi * vf * 2 * t)   # 2nd harmonic
              + 0.8 * va * rng.standard_normal(N))            # broadband road noise

    # Road surface noise — class-specific amplitude, shared structure
    # human and drone: rn=0.0 → no road noise (not in contact with road surface)
    # car and truck:   rn=3.0 → equal road noise (same road, same dashboard coupling)
    # This forces the classifier to discriminate car vs truck using vib_freq and
    # v_mean rather than heading_rms (which is now equally noisy for both).
    if rn > 0.0:
        _bump_amp  = 2.0    # periodic bump amplitude [deg/s]
        _bump_freq = 1.2    # bump frequency [Hz]
        _sway_amp  = 1.0    # low-frequency sway [deg/s]
        _sway_freq = 0.3    # sway frequency [Hz]
        ph_bump = rng.uniform(0, 2 * np.pi)
        ph_sway = rng.uniform(0, 2 * np.pi)
        road_noise = (
            rn * rng.standard_normal(N)
            + _bump_amp * np.sin(2 * np.pi * _bump_freq * t + ph_bump)
            + _sway_amp * np.sin(2 * np.pi * _sway_freq * t + ph_sway)
        )
    else:
        road_noise = np.zeros(N)

    # Yaw rate: driving dynamics + road surface noise
    omega_z = np.deg2rad(hs) * rng.standard_normal(N) + np.deg2rad(road_noise)

    # Heading rate: driving dynamics + road surface noise
    heading_rate = hs * rng.standard_normal(N) + road_noise

    return {
        'obj_class':    obj_class,
        't':            t,
        'velocity':     v,
        'a_long':       a_long,
        'a_lat':        a_lat,
        'a_vert':       a_vert,
        'omega_z':      omega_z,
        'heading_rate': heading_rate,
        'fs':           fs,
    }


def generate_dataset(n_per_class: int = 300,
                      classes: list = None,
                      duration: float = 5.0,
                      fs: float = 100.0,
                      seed_offset: int = 0,
                      extractor=None) -> tuple:
    """
    Generate a labeled dataset for training/evaluation.

    Parameters
    ----------
    n_per_class : samples per class
    classes     : list of class names (default: all 5)
    duration    : window duration [s]
    fs          : IMU sample rate [Hz]
    seed_offset : added to seed for train/test split
    extractor   : SGIFeatureExtractor instance (auto-created if None)

    Returns
    -------
    X : np.ndarray (N_total, 14)
    y : np.ndarray (N_total,) of str labels
    """
    from sgi._internal.features import SGIFeatureExtractor

    if classes is None:
        classes = DEFAULT_CLASSES
    if extractor is None:
        extractor = SGIFeatureExtractor(fs=fs)

    X, y = [], []
    for i, cls in enumerate(classes):
        for j in range(n_per_class):
            w    = generate_window(cls, duration=duration, fs=fs,
                                   seed=i * 10000 + j + seed_offset)
            feat = extractor.extract(w)
            X.append(feat)
            y.append(cls)

    return np.array(X, dtype=np.float32), np.array(y)
