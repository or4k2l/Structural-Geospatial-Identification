"""
sgi._internal.physics
─────────────────────
Physical constants and signal amplitude calculations for SGI.

SGI-Light:  GPS+IMU motion signatures  (deployed, this module)
SGI-Full:   Relativistic K-field       (theoretical, documented here)

Author: Yahya Akbay | 2025
"""

import numpy as np

# ── Physical constants (SI) ───────────────────────────────────────────────────
G_SI   = 6.674e-11   # m³/(kg·s²)
C_SI   = 2.998e8     # m/s
M_E    = 5.972e24    # kg  (Earth mass)
R_E    = 6.371e6     # m   (Earth mean radius)
N_A    = 6.022e23    # Avogadro's number

# Derived
R_S_EARTH = 2 * G_SI * M_E / C_SI**2       # Schwarzschild radius of Earth [m]
EPSILON   = R_S_EARTH / R_E                 # Relativistic surface correction ~1.39e-9

# ── SGI-Full: K-field amplitude (SI) ─────────────────────────────────────────

def k_amplitude(mass_kg: float, velocity_ms: float,
                 detection_range_m: float = 10.0) -> float:
    """
    Order-of-magnitude estimate of the K-field amplitude [m⁻¹]
    for a single object in Earth's gravitational field.

    Signal = (G/c⁴) · m · v² / r · ε

    Parameters
    ----------
    mass_kg           : object mass [kg]
    velocity_ms       : object velocity [m/s]
    detection_range_m : distance to detector [m]

    Returns
    -------
    K amplitude [m⁻¹]

    Notes
    -----
    This is a theoretical estimate. The absolute signal for Earth-surface
    objects is ~10⁻⁴⁴ m⁻¹ — approximately 35 orders of magnitude below
    any known detector. See SGI v3.0 and the Collective Field notebook
    for the full gap analysis.
    """
    return (G_SI / C_SI**4) * mass_kg * velocity_ms**2 / detection_range_m * EPSILON


def k_to_delta_g(k_amplitude_m: float,
                  detection_range_m: float = 10.0) -> float:
    """
    Convert K-field amplitude to effective gravity perturbation [m/s²].
    δg ≈ K · c² / r
    """
    return k_amplitude_m / detection_range_m * C_SI**2


def collective_k(k_single: float, n_objects: int,
                  mode: str = 'incoherent') -> float:
    """
    Collective K-field from N identical objects.

    Parameters
    ----------
    k_single  : single-object K amplitude [m⁻¹]
    n_objects : number of objects
    mode      : 'coherent'   → N × K  (upper bound, requires phase alignment)
                'incoherent' → √N × K (realistic, random phases)

    Returns
    -------
    Collective K amplitude [m⁻¹]
    """
    if mode == 'coherent':
        return k_single * n_objects
    elif mode == 'incoherent':
        return k_single * np.sqrt(n_objects)
    else:
        raise ValueError(f"mode must be 'coherent' or 'incoherent', got '{mode}'")


def gap_to_detector(mass_kg: float, velocity_ms: float,
                     detector_sensitivity_ms2: float = 1e-9,
                     detection_range_m: float = 10.0) -> dict:
    """
    Compute the technology gap between a single-object SGI-Full signal
    and a target detector sensitivity.

    Returns
    -------
    dict with keys:
        k_amplitude    : K-field amplitude [m⁻¹]
        delta_g        : effective gravity perturbation [m/s²]
        gap_orders     : orders of magnitude gap to detector
        n_coherent     : N objects needed (coherent) to reach detector
        n_incoherent   : N objects needed (incoherent) to reach detector
    """
    K   = k_amplitude(mass_kg, velocity_ms, detection_range_m)
    dg  = k_to_delta_g(K, detection_range_m)

    if dg >= detector_sensitivity_ms2:
        gap    = 0.0
        n_coh  = 1
        n_inc  = 1
    else:
        gap   = np.log10(detector_sensitivity_ms2 / dg)
        n_coh = detector_sensitivity_ms2 / dg
        n_inc = (detector_sensitivity_ms2 / dg) ** 2

    return {
        'k_amplitude':   K,
        'delta_g':       dg,
        'gap_orders':    gap,
        'n_coherent':    n_coh,
        'n_incoherent':  n_inc,
    }


# ── Known detector sensitivities [m/s²] ──────────────────────────────────────
DETECTOR_SENSITIVITIES = {
    'smartphone_imu':      1e-2,
    'mems_accelerometer':  1e-5,
    'quantum_gravimeter':  1e-9,
    'grace_fo':            1e-10,
    'atom_interferometer': 1e-15,
    'ligo':                1e-23,
}

# ── Characteristic object parameters ─────────────────────────────────────────
OBJECT_PARAMETERS = {
    'human':     {'mass': 80,       'velocity': 1.4,   'vib_freq': 1.8},
    'bicycle':   {'mass': 90,       'velocity': 4.5,   'vib_freq': 2.5},
    'car':       {'mass': 1500,     'velocity': 13.9,  'vib_freq': 5.0},
    'truck':     {'mass': 15000,    'velocity': 22.2,  'vib_freq': 8.0},
    'drone':     {'mass': 1.5,      'velocity': 10.0,  'vib_freq': 50.0},
    'airplane':  {'mass': 70000,    'velocity': 250.0, 'vib_freq': 0.0},
    'submarine': {'mass': 8000000,  'velocity': 10.0,  'vib_freq': 0.0},
}
