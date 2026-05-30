# AirWrite(In progress)

**Gesture-Based Touchless Input for Cursor Control and Continuous Air Handwriting**

AirWrite is a webcam-based HCI system that combines real-time cursor control, gesture clicking, continuous air handwriting, and OS-level text injection — built for accessibility and FYP demonstration.

## Features

- **Cursor mode** — move cursor (index finger), left click (pinch), right click (hold pinch 1s), scroll (two-finger pinch + drag)
- **Write mode** — draw words in the air (pinch + move), progressive CNN recognition, commit after 1.2s pause
- **Mode switch** — open palm hold (~0.8s)
- **Wipe gesture** — clear current word or delete last injected word
- **HUD** — mode, FPS, gesture, live/stable prediction, pen state

## Requirements

- Python 3.11+
- Windows/Linux
- HD webcam
- 8 GB RAM minimum

## Setup

```bash
cd AirWrite
pip install -r requirements.txt
```

## Train the CNN (EMNIST Balanced)

```bash
python -m ml.training.train_cnn --epochs 5
```

Weights are saved to `ml/models/emnist_cnn.pt`.

## Run

```bash
python full_pipeline.py
```

- **Q** or **Esc** — quit  
- **Backspace** — delete character (write mode)  
- **Open palm hold** — switch Cursor ↔ Write  

## Project structure

```
AirWrite/
├── core/
│   ├── gesture_engine.py
│   ├── stroke_engine.py
│   ├── cursor_engine.py
│   └── segmentation_engine.py
├── ml/
│   ├── models/cnn.py
│   ├── training/train_cnn.py
│   └── inference/predictor.py
├── utils/
│   ├── smoothing.py
│   ├── hud.py
│   └── helpers.py
├── tests/
├── docs/
├── config.py
├── full_pipeline.py
└── README.md
```

## Documentation

See `Airwrite Master Project Documentation.pdf` for full architecture, requirements, and roadmap.

