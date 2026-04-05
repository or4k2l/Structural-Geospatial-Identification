# Changelog

## [1.0.0] — May 2025

### Initial Release

**SGI-Light (deployed)**
- `SGILightClassifier` — Random Forest pipeline, 5 classes
- `SGIFeatureExtractor` — 12 physically motivated features
- `generate_window / generate_dataset` — synthetic GPS+IMU data
- Save / load trained models via joblib
- Input: dict, numpy (N,6), pandas DataFrame
- 36 tests, 100% passing

**SGI-Full (theoretical, physics API)**
- `k_amplitude()` — K-field amplitude in SI units
- `k_to_delta_g()` — convert to gravity perturbation
- `collective_k()` — N-object superposition (coherent / incoherent)
- `gap_to_detector()` — technology gap analysis
- `DETECTOR_SENSITIVITIES` — reference sensitivity table
- `OBJECT_PARAMETERS` — characteristic mass/velocity per class
- `EPSILON` — Earth surface relativistic correction (~1.39e-9)

**Pretrained model**
- `sgi/models/sgi_light_v1.pkl` — trained on 2500 samples, CV 100%

**Companion Kaggle Notebooks**
- SGI v3.0 — full theoretical framework
- SGI Collective Field — gap analysis
- SGI-Light Prototype — hardware spec and open hardware call
