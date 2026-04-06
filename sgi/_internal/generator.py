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
# (v_mean, v_std, a_long, a_lat, vib_freq, vib_amp, heading_std)
# Sources:
#   Human gait:      Menz et al. (2003)          → ~1.8 Hz
#   Bicycle cadence: Wilson (2004)                → ~2.5 Hz
#   Car engine idle: 4-cyl, 600 RPM              → ~5 Hz
#   Truck diesel:    6-cyl, 800 RPM              → ~8 Hz
#   Drone rotor:     DJI Mini series specs        → ~50 Hz
#   car/truck separation: multi-feature (vib_freq + a_long + heading_std + v_std)
#   Calibrated against PVS dataset (Menegazzo, 2020) — real GPS validation

MOTION_PARAMS = {
    'human':   (1.4,  0.30, 0.05, 0.03, 1.8,  0.08, 15.0),  # unchanged
    'bicycle': (4.5,  0.80, 0.10, 0.05, 2.5,  0.04,  8.0),  # unchanged
    'car':     (13.9, 3.50, 0.80, 0.40, 5.0,  0.02,  5.0),  # v_std: 2.00→3.50 (city stop-and-go)
    'truck':   (11.1, 1.20, 1.40, 0.25, 8.0,  0.08,  1.5),  # v_mean: 8.3→11.1 (40km/h), v_std: 1.50→1.20, a_long: 0.40→1.40, a_lat: 0.20→0.25, heading_std: 2.0→1.5
    'drone':   (8.0,  1.50, 0.60, 0.60, 50.0, 0.03, 25.0),  # unchanged
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
    """
    if obj_class not in MOTION_PARAMS:
        raise ValueError(f"Unknown class '{obj_class}'. "
                         f"Available: {list(MOTION_PARAMS.keys())}")

    rng = np.random.default_rng(seed)

    vm, vs, al_s, at_s, vf, va, hs = MOTION_PARAMS[obj_class]
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

    # Yaw rate
    omega_z = np.deg2rad(hs) * rng.standard_normal(N)

    # Heading rate
    heading_rate = hs * rng.standard_normal(N)

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
    X : np.ndarray (N_total, 11)
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
