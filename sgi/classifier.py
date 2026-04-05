"""
sgi.classifier
──────────────
SGILightClassifier — the core SGI-Light prediction engine.

Author: Yahya Akbay | 2025
"""

import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from sgi._internal.features import SGIFeatureExtractor, FEATURE_NAMES
from sgi._internal.generator import generate_dataset, generate_window, DEFAULT_CLASSES

_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
_DEFAULT_MODEL_PATH = os.path.join(_MODEL_DIR, 'sgi_light_v1.pkl')


class SGILightClassifier:
    """
    SGI-Light object classifier from GPS+IMU motion data.

    Classifies objects into 5 classes based on their dynamic motion
    signature: human, bicycle, car, truck, drone.

    Usage
    -----
    Quick start:
        >>> import sgi
        >>> result = sgi.predict(sensor_data)

    Full API:
        >>> clf = SGILightClassifier()
        >>> clf.train()
        >>> result = clf.predict(window)
        >>> proba  = clf.predict_proba(window)
        >>> clf.save()

    Parameters
    ----------
    classes : list of str
        Object classes to classify. Default: ['human','bicycle','car','truck','drone']
    fs : float
        IMU sample rate [Hz]. Default: 100
    """

    VERSION = '1.0.0'

    def __init__(self, classes=None, fs: float = 100.0):
        self.classes   = classes or DEFAULT_CLASSES
        self.fs        = fs
        self.extractor = SGIFeatureExtractor(fs=fs)
        self.pipeline  = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ))
        ])
        self.is_trained    = False
        self.cv_scores_    = None
        self.feature_importances_ = None
        self._n_train      = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, n_per_class: int = 500, verbose: bool = True) -> 'SGILightClassifier':
        """
        Generate synthetic training data and fit the classifier.

        Parameters
        ----------
        n_per_class : int   — samples per class (default 500)
        verbose     : bool  — print progress

        Returns
        -------
        self (for chaining)
        """
        if verbose:
            total = n_per_class * len(self.classes)
            print(f"SGI-Light: generating {total} training samples...")

        X, y = generate_dataset(n_per_class=n_per_class,
                                  classes=self.classes,
                                  fs=self.fs)
        self.pipeline.fit(X, y)
        self.is_trained = True
        self._n_train   = len(X)

        rf = self.pipeline.named_steps['clf']
        self.feature_importances_ = dict(
            zip(FEATURE_NAMES, rf.feature_importances_)
        )

        if verbose:
            print(f"SGI-Light: training complete ({self._n_train} samples).")

        return self

    def evaluate(self, n_per_class: int = 100,
                  cv: int = 5, verbose: bool = True) -> np.ndarray:
        """
        Stratified k-fold cross-validation on held-out synthetic data.

        Returns
        -------
        np.ndarray of shape (cv,) with accuracy scores
        """
        X_eval, y_eval = generate_dataset(
            n_per_class=n_per_class,
            classes=self.classes,
            fs=self.fs,
            seed_offset=99999,
        )
        skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.pipeline, X_eval, y_eval,
                                  cv=skf, scoring='accuracy')
        self.cv_scores_ = scores
        if verbose:
            print(f"SGI-Light CV ({cv}-fold): "
                  f"{scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
        return scores

    # ── Inference ─────────────────────────────────────────────────────────────

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError(
                "Classifier not trained. Call .train() or .load() first."
            )

    def predict(self, data) -> str:
        """
        Classify one IMU window.

        Parameters
        ----------
        data : dict | np.ndarray (N,6) | pd.DataFrame
            Sensor data for one window.

        Returns
        -------
        str : predicted class label
        """
        self._check_trained()
        feat = self._to_features(data)
        return self.pipeline.predict(feat.reshape(1, -1))[0]

    def predict_proba(self, data) -> dict:
        """
        Class probabilities for one IMU window.

        Returns
        -------
        dict mapping class name → probability
        """
        self._check_trained()
        feat  = self._to_features(data)
        proba = self.pipeline.predict_proba(feat.reshape(1, -1))[0]
        return dict(zip(self.pipeline.classes_, proba))

    def predict_batch(self, windows: list) -> list:
        """
        Classify a list of windows. Returns list of class labels.
        """
        self._check_trained()
        feats = np.array([self._to_features(w) for w in windows])
        return list(self.pipeline.predict(feats))

    def _to_features(self, data) -> np.ndarray:
        """Convert various input formats to feature vector."""
        if isinstance(data, dict):
            return self.extractor.extract(data)
        elif hasattr(data, 'values'):
            # pandas DataFrame
            return self.extractor.extract_dataframe(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1 and len(data) == len(FEATURE_NAMES):
                return data.astype(np.float32)
            elif data.ndim == 2 and data.shape[1] == 6:
                return self.extractor.extract_array(data)
            else:
                raise ValueError(
                    f"numpy array must be shape ({len(FEATURE_NAMES)},) [features] "
                    "or (N,6) [raw sensor columns]"
                )
        else:
            raise TypeError(
                f"Unsupported input type: {type(data)}. "
                "Use dict (window), DataFrame, or numpy array."
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = None) -> str:
        """
        Save trained model to disk.

        Parameters
        ----------
        path : str — file path (default: sgi/models/sgi_light_v1.pkl)

        Returns
        -------
        str : saved path
        """
        if not _JOBLIB_AVAILABLE:
            raise ImportError("joblib required for model serialization. "
                              "pip install joblib")
        self._check_trained()
        path = path or _DEFAULT_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'pipeline':             self.pipeline,
            'classes':              self.classes,
            'fs':                   self.fs,
            'cv_scores':            self.cv_scores_,
            'feature_importances':  self.feature_importances_,
            'n_train':              self._n_train,
            'version':              self.VERSION,
        }, path)
        print(f"SGI-Light: model saved → {path}")
        return path

    @classmethod
    def load(cls, path: str = None) -> 'SGILightClassifier':
        """
        Load a trained model from disk.

        Parameters
        ----------
        path : str — file path (default: bundled model)

        Returns
        -------
        SGILightClassifier (trained)
        """
        if not _JOBLIB_AVAILABLE:
            raise ImportError("joblib required. pip install joblib")
        path = path or _DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No model found at '{path}'. "
                "Call SGILightClassifier().train().save() first, "
                "or use sgi.load()."
            )
        data = joblib.load(path)
        obj  = cls(classes=data['classes'], fs=data['fs'])
        obj.pipeline              = data['pipeline']
        obj.is_trained            = True
        obj.cv_scores_            = data.get('cv_scores')
        obj.feature_importances_  = data.get('feature_importances')
        obj._n_train              = data.get('n_train', 0)
        print(f"SGI-Light: model loaded ← {path}  "
              f"(v{data.get('version','?')})")
        return obj

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self):
        """Print a human-readable performance summary."""
        print("SGI-Light Classifier")
        print("═" * 50)
        print(f"  Version  : {self.VERSION}")
        print(f"  Classes  : {self.classes}")
        print(f"  Features : {len(FEATURE_NAMES)}")
        print(f"  IMU rate : {self.fs} Hz")
        if self.is_trained:
            print(f"  Training : {self._n_train} samples")
        if self.cv_scores_ is not None:
            print(f"  CV Acc   : {self.cv_scores_.mean()*100:.1f}% "
                  f"± {self.cv_scores_.std()*100:.1f}%")
        if self.feature_importances_:
            print()
            print("  Top features:")
            sorted_fi = sorted(self.feature_importances_.items(),
                                key=lambda x: x[1], reverse=True)
            for name, imp in sorted_fi[:5]:
                bar = '█' * int(imp * 40)
                print(f"    {name:<18} {imp:.3f}  {bar}")
