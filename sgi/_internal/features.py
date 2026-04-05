"""
sgi._internal.features
──────────────────────
Feature extraction from GPS+IMU sensor windows.

11 physically motivated features, all computable on edge hardware
(ESP32, Raspberry Pi).

Author: Yahya Akbay | 2025
"""

import warnings

import numpy as np
from scipy.signal import welch

# ── Default hardware parameters ───────────────────────────────────────────────
DEFAULT_IMU_HZ    = 100   # ICM-42688-P typical rate
DEFAULT_GPS_HZ    = 10    # u-blox NEO-M9N max rate (documented hardware constant, not used in computation)
DEFAULT_WINDOW_S  = 5     # classification window [seconds] (documented hardware constant, not used in computation)

FEATURE_NAMES = [
    'v_mean',        # mean forward velocity [m/s]
    'v_std',         # velocity std [m/s]
    'v_max',         # max velocity [m/s]
    'a_long_rms',    # longitudinal acceleration RMS [m/s²]
    'a_lat_rms',     # lateral acceleration RMS [m/s²]
    'jerk_rms',      # jerk RMS [m/s³]
    'vib_freq',      # dominant vertical vibration frequency [Hz]
    'vib_amp',       # vertical vibration PSD peak [m²/s⁴/Hz]
    'heading_rms',   # heading rate RMS [deg/s]
    'omega_rms',     # yaw rate RMS [rad/s]
    'ke_proxy',      # kinetic energy proxy: mean(v²) [m²/s²]
]

N_FEATURES = len(FEATURE_NAMES)


class SGIFeatureExtractor:
    """
    Extract the SGI-Light feature vector from one GPS+IMU window.

    Parameters
    ----------
    fs : float
        IMU sample rate [Hz]. Default: 100 Hz (ICM-42688-P)

    Usage
    -----
    >>> ext = SGIFeatureExtractor()
    >>> features = ext.extract(window_dict)   # shape (12,)
    >>> features = ext.extract_dataframe(df)  # from pandas DataFrame
    >>> features = ext.extract_array(arr)     # from numpy array (N×6)
    """

    def __init__(self, fs: float = DEFAULT_IMU_HZ):
        self.fs = fs

    def extract(self, window: dict) -> np.ndarray:
        """
        Extract features from a window dict.

        Expected keys:
            velocity      : np.ndarray [m/s]  — real GPS-derived speed (required
                            for vehicle classification; see GPS Velocity note below)
            a_long        : np.ndarray [m/s²]  (longitudinal)
            a_lat         : np.ndarray [m/s²]  (lateral)
            a_vert        : np.ndarray [m/s²]  (vertical, gravity-removed)
            omega_z       : np.ndarray [rad/s] (yaw rate)
            heading_rate  : np.ndarray [deg/s]
            fs            : float (optional, overrides self.fs)

        Returns
        -------
        np.ndarray of shape (11,), dtype float32

        Note: GPS Velocity
        ------------------
        SGI is a GPS+IMU classifier. The ``velocity`` field must contain real
        GPS-derived speed in m/s. Using integrated acceleration as a velocity
        proxy will yield poor accuracy for vehicle classes (car, truck, bicycle).
        """
        v   = np.asarray(window['velocity'],     dtype=float)
        al  = np.asarray(window['a_long'],       dtype=float)
        at  = np.asarray(window['a_lat'],        dtype=float)
        av  = np.asarray(window['a_vert'],       dtype=float)
        oz  = np.asarray(window['omega_z'],      dtype=float)
        hr  = np.asarray(window['heading_rate'], dtype=float)
        fs  = float(window.get('fs', self.fs))

        v_mean_val = float(np.mean(v))
        v_std_val  = float(np.std(v))
        if v_std_val < 0.1 and v_mean_val < 0.5:
            warnings.warn(
                f"SGIFeatureExtractor: 'velocity' appears to be constant or "
                f"near-zero (mean={v_mean_val:.3f}, std={v_std_val:.3f}). "
                "SGI requires real GPS-derived speed for vehicle classification "
                "(car, truck, bicycle). Using integrated acceleration as velocity "
                "proxy will result in poor accuracy for non-human classes.",
                UserWarning,
                stacklevel=2,
            )

        # Velocity
        v_mean = v_mean_val
        v_std  = v_std_val
        v_max  = float(np.max(v))

        # Acceleration
        a_long_rms = float(np.sqrt(np.mean(al**2)))
        a_lat_rms  = float(np.sqrt(np.mean(at**2)))
        jerk_rms   = float(np.sqrt(np.mean(np.diff(al)**2)) * fs)

        # Vibration (PSD of vertical acceleration)
        nperseg    = min(256, max(4, len(av) // 4))
        freqs, psd = welch(av, fs=fs, nperseg=nperseg)
        peak_idx   = int(np.argmax(psd))
        vib_freq   = float(freqs[peak_idx])
        vib_amp    = float(psd[peak_idx])

        # Heading / yaw
        heading_rms = float(np.sqrt(np.mean(hr**2)))
        omega_rms   = float(np.sqrt(np.mean(oz**2)))

        # Kinetic energy proxy
        ke_proxy = float(np.mean(v**2))

        return np.array([
            v_mean, v_std, v_max,
            a_long_rms, a_lat_rms, jerk_rms,
            vib_freq, vib_amp,
            heading_rms, omega_rms,
            ke_proxy,
        ], dtype=np.float32)

    def extract_batch(self, windows: list) -> np.ndarray:
        """Extract features from a list of window dicts. Returns (N, 11)."""
        return np.array([self.extract(w) for w in windows], dtype=np.float32)

    def extract_dataframe(self, df) -> np.ndarray:
        """
        Extract features from a pandas DataFrame.

        Expected columns: velocity, a_long, a_lat, a_vert, omega_z, heading_rate
        """
        window = {
            'velocity':     df['velocity'].values,
            'a_long':       df['a_long'].values,
            'a_lat':        df['a_lat'].values,
            'a_vert':       df['a_vert'].values,
            'omega_z':      df['omega_z'].values,
            'heading_rate': df['heading_rate'].values,
            'fs':           self.fs,
        }
        return self.extract(window)

    def extract_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Extract features from a numpy array of shape (N, 6).

        Column order: [velocity, a_long, a_lat, a_vert, omega_z, heading_rate]
        """
        if not (arr.ndim == 2 and arr.shape[1] == 6):
            raise ValueError(
                "Array must be shape (N, 6): [v, a_long, a_lat, a_vert, omega_z, heading_rate]"
            )
        window = {
            'velocity':     arr[:, 0],
            'a_long':       arr[:, 1],
            'a_lat':        arr[:, 2],
            'a_vert':       arr[:, 3],
            'omega_z':      arr[:, 4],
            'heading_rate': arr[:, 5],
            'fs':           self.fs,
        }
        return self.extract(window)

    @property
    def feature_names(self):
        return FEATURE_NAMES

    @property
    def n_features(self):
        return N_FEATURES
