# sgi-machine

**Structural Geospatial Identification — Python Package**

> *WHERE IS WHAT?*  
> GPS tells us **where**. SGI tells us **what**. Together: passive object identification, no camera, no radar.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-SGI%20v3.0-20BEFF.svg)](https://www.kaggle.com/yahyaakbay)

---

## Install

```bash
pip install sgi-machine
```

Or from source:

```bash
git clone https://github.com/or4k2l/Structural-Geospatial-Identification
cd Structural-Geospatial-Identification
pip install -e .
```

---

## Quick Start

```python
import sgi

# Train on synthetic data (takes ~5 seconds)
clf = sgi.train()

# Classify one sensor window
result = clf.predict(sensor_window)   # → 'truck'

# With confidence scores
proba = clf.predict_proba(sensor_window)
# → {'human': 0.01, 'bicycle': 0.01, 'car': 0.03, 'truck': 0.94, 'drone': 0.01}

# Save / load trained model
clf.save()
clf = sgi.load()

# Package info + physical constants
sgi.info()
```

---

## Input Format

`sensor_window` can be any of:

**Dict (from hardware parser):**
```python
window = {
    'velocity':     np.array([...]),   # forward speed [m/s]
    'a_long':       np.array([...]),   # longitudinal accel [m/s²]
    'a_lat':        np.array([...]),   # lateral accel [m/s²]
    'a_vert':       np.array([...]),   # vertical accel, gravity-removed [m/s²]
    'omega_z':      np.array([...]),   # yaw rate [rad/s]
    'heading_rate': np.array([...]),   # heading change [deg/s]
    'fs':           100.0,             # sample rate [Hz]
}
```

**NumPy array `(N, 6)`:**
```python
# columns: [velocity, a_long, a_lat, a_vert, omega_z, heading_rate]
arr = sensor_data[:, :6]
clf.predict(arr)
```

**Pandas DataFrame** with the above column names.

> **⚠️ GPS Velocity Required**  
> `velocity` must be real GPS-derived speed in m/s. SGI is a **GPS + IMU classifier**.  
> Passing a near-zero or constant velocity array will trigger a `UserWarning` and will
> result in poor classification accuracy for all vehicle classes.

---

## Known Limitations

| Limitation | Detail |
|-----------|--------|
| **GPS speed is mandatory** | `velocity` must come from real GPS (e.g. u-blox NEO-M9N). Integrated-acceleration proxies degrade vehicle-class accuracy to near-zero. |
| **IMU-only mode** | Without real GPS speed, only the `human` class is reliably classified (~93%). All vehicle classes (`car`, `truck`, `bicycle`) collapse. |
| **Synthetic training** | The classifier is trained on synthetic data. Cross-domain performance on real-world data depends on GPS availability. |

### Real-World Validation — Collecty Dataset

Experiment: SGI-Machine classifier trained on synthetic data, evaluated against the
[Collecty dataset](https://doi.org/10.1016/j.dib.2023.109481)
(Zagreb, Croatia, 100 Hz, 242 hours of labelled transport data) **without** real GPS speed.

| Class | Accuracy | Note |
|-------|----------|------|
| `human` | 93.4% | ✅ IMU signal alone is sufficient |
| `truck` | 12.4% | ⚠️ Requires GPS speed |
| `car` | 1.0% | ❌ Requires GPS speed |
| `bicycle` | 0.0% | ❌ Requires GPS speed |

**Conclusion:** SGI achieves strong human-class accuracy from IMU alone, but reliable
vehicle classification (`car`, `truck`, `bicycle`) requires real GPS-derived speed in
the `velocity` field.

> Erdelić, M., Erdelić, T., & Carić, T. (2023). Dataset for multimodal transport analytics
> of smartphone users — Collecty. *Data in Brief*, 109481.
> <https://doi.org/10.1016/j.dib.2023.109481>

---

## Object Classes

| Class | Mass | Velocity | Key signature |
|-------|------|----------|---------------|
| `human` | 80 kg | 1.4 m/s | Gait ~1.8 Hz |
| `bicycle` | 90 kg | 4.5 m/s | Cadence ~2.5 Hz |
| `car` | 1500 kg | 13.9 m/s | Engine ~5 Hz |
| `truck` | 15000 kg | 22.2 m/s | Diesel ~8 Hz |
| `drone` | 1.5 kg | 8.0 m/s | Rotor ~50 Hz |

---

## Package Structure

```
sgi/
├── __init__.py          ← public API: train, load, predict, predict_proba, info
├── classifier.py        ← SGILightClassifier
├── _internal/
│   ├── features.py      ← SGIFeatureExtractor (11 features)
│   ├── generator.py     ← synthetic GPS+IMU data generator
│   └── physics.py       ← SGI-Full: K-field, gap analysis (theoretical)
└── models/
    └── sgi_light_v1.pkl ← bundled pretrained model (after sgi.train().save())
tests/
└── test_sgi.py          ← 37 tests, all passing
```

---

## Hardware Interface

This package is designed to run on:

| Platform | Role |
|----------|------|
| Raspberry Pi Zero 2W | Full Python pipeline, WiFi streaming |
| ESP32-S3 | Feature extraction in C++, WiFi output |
| Any Linux device | Full pipeline |

**Sensor requirements:**
- GPS: u-blox NEO-M9N (10 Hz) or equivalent
- IMU: ICM-42688-P or MPU-6050 (100 Hz)

**Total hardware cost: ~69€**

---

## SGI Spectrum

```
SGI-Light  (this package)   GPS+IMU+ML → deployed today
SGI-Medium (planned)        Quantum gravimetry → research
SGI-Full   (theoretical)    Relativistic K-field → 35-order gap
```

### Physics API (SGI-Full, theoretical)

```python
import sgi

# K-field amplitude for a single object
K = sgi.k_amplitude(mass_kg=80, velocity_ms=1.4)
# → 1.80e-52 m⁻¹  (unmeasurable with current technology)

# Gap analysis
result = sgi.gap_to_detector(80, 1.4, detector_sensitivity_ms2=1e-9)
result['gap_orders']    # → 26.8 orders to quantum gravimeter
result['n_coherent']    # → 6.17e+26 objects needed (coherent)

# Collective field
K_collective = sgi.collective_k(K, n_objects=1000, mode='incoherent')

# Earth relativistic correction (same as GPS correction)
sgi.EPSILON   # → 1.39e-9
```

See [SGI Collective Field notebook](https://www.kaggle.com/yahyaakbay/sgi-collective-field) for the full gap analysis.

---

## Companion Notebooks (Kaggle)

| Notebook | Description |
|----------|-------------|
| [SGI v3.0](https://www.kaggle.com/yahyaakbay) | Full theoretical framework |
| [SGI Collective Field](https://www.kaggle.com/yahyaakbay) | Gap analysis: can N objects bridge 35 orders? |
| [SGI-Light Prototype](https://www.kaggle.com/yahyaakbay) | Hardware spec, open hardware call |

---

## 🔧 Hardware Partner Wanted

The software is ready. The hardware is not.

If you can build a GPS+IMU prototype on a Raspberry Pi or ESP32 — let's build this together.

**You bring:** hardware, soldering iron, 3D printer  
**I bring:** software, documentation, physics  
**We build:** the first open SGI-Light prototype

→ Open an issue or comment on the Kaggle notebook.

---

## Citation

```bibtex
@software{akbay2025sgi,
  author  = {Akbay, Yahya},
  title   = {sgi-machine: Structural Geospatial Identification},
  year    = {2025},
  url     = {https://github.com/or4k2l/Structural-Geospatial-Identification},
  version = {1.0.0}
}
```

---

## License

MIT © 2025 Yahya Akbay
