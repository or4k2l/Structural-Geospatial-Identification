"""
sgi — Structural Geospatial Identification
==========================================
GPS+IMU based object classification.

Public API
----------
High-level convenience functions:
    train(n_per_class, verbose)  → SGILightClassifier
    load(path)                   → SGILightClassifier
    predict(data)                → str
    predict_proba(data)          → dict
    info()                       → None

Core classes / functions re-exported for power users:
    SGILightClassifier
    SGIFeatureExtractor
    generate_window, generate_dataset
    k_amplitude, k_to_delta_g, collective_k, gap_to_detector
    EPSILON, DETECTOR_SENSITIVITIES, OBJECT_PARAMETERS
    DEFAULT_CLASSES, FEATURE_NAMES

Author: Yahya Akbay | 2025
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__  = "Yahya Akbay"

# ── Core classes ──────────────────────────────────────────────────────────────
from sgi.classifier import SGILightClassifier

# ── Internal sub-modules re-exported ─────────────────────────────────────────
from sgi._internal.features  import SGIFeatureExtractor, FEATURE_NAMES
from sgi._internal.generator import generate_window, generate_dataset, DEFAULT_CLASSES
from sgi._internal.physics   import (
    k_amplitude, k_to_delta_g, collective_k, gap_to_detector,
    EPSILON, DETECTOR_SENSITIVITIES, OBJECT_PARAMETERS,
)

# ── Module-level lazy state (one shared classifier) ───────────────────────────
_clf: SGILightClassifier | None = None


def _get_or_train() -> SGILightClassifier:
    """Return the module-level classifier, training it if necessary."""
    global _clf
    if _clf is None or not _clf.is_trained:
        _clf = SGILightClassifier()
        _clf.train(verbose=False)
    return _clf


# ── Public convenience API ────────────────────────────────────────────────────

def train(n_per_class: int = 500, verbose: bool = True) -> SGILightClassifier:
    """
    Train a new SGILightClassifier on synthetic data.

    Parameters
    ----------
    n_per_class : int  — training samples per class (default 500)
    verbose     : bool — print progress

    Returns
    -------
    SGILightClassifier (trained)
    """
    global _clf
    _clf = SGILightClassifier()
    _clf.train(n_per_class=n_per_class, verbose=verbose)
    return _clf


def load(path: str = None) -> SGILightClassifier:
    """
    Load a trained model from disk (default: bundled pretrained model).

    Parameters
    ----------
    path : str — file path (default: sgi/models/sgi_light_v1.pkl)

    Returns
    -------
    SGILightClassifier (trained)
    """
    global _clf
    _clf = SGILightClassifier.load(path)
    return _clf


def predict(data) -> str:
    """
    Classify one sensor window using the module-level classifier.
    Trains automatically on first call if no model is loaded.

    Parameters
    ----------
    data : dict | np.ndarray (N,6) | pd.DataFrame

    Returns
    -------
    str : predicted class label
    """
    return _get_or_train().predict(data)


def predict_proba(data) -> dict:
    """
    Class probabilities for one sensor window.

    Returns
    -------
    dict mapping class name → probability
    """
    return _get_or_train().predict_proba(data)


def info() -> None:
    """Print package information, version, and physical constants."""
    from sgi._internal.physics import EPSILON, G_SI, C_SI

    print(f"sgi-machine  v{__version__}  — Structural Geospatial Identification")
    print("=" * 60)
    print(f"  Classes   : {DEFAULT_CLASSES}")
    print(f"  Features  : {len(FEATURE_NAMES)}  ({', '.join(FEATURE_NAMES)})")
    print()
    print("  Physical constants (SI):")
    print(f"    G        = {G_SI:.3e}  m³/(kg·s²)")
    print(f"    c        = {C_SI:.3e}  m/s")
    print(f"    ε (EPSILON) = {EPSILON:.3e}  (Earth surface relativistic correction)")
    print()
    print("  SGI Spectrum:")
    print("    SGI-Light  (this package)  GPS+IMU+ML  → deployed today")
    print("    SGI-Medium (planned)       Quantum gravimetry → research")
    print("    SGI-Full   (theoretical)   Relativistic K-field → 35-order gap")


__all__ = [
    # version
    "__version__",
    # convenience API
    "train", "load", "predict", "predict_proba", "info",
    # classes
    "SGILightClassifier", "SGIFeatureExtractor",
    # generator
    "generate_window", "generate_dataset",
    # physics
    "k_amplitude", "k_to_delta_g", "collective_k", "gap_to_detector",
    # constants
    "EPSILON", "DETECTOR_SENSITIVITIES", "OBJECT_PARAMETERS",
    "DEFAULT_CLASSES", "FEATURE_NAMES",
]
