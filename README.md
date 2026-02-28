# Edge-RL Tomato Digital Twin

An end-to-end framework for autonomous tomato ripening control, integrating direct pixel-based feature extraction, Reinforcement Learning (DQN), and a Digital Twin visualization system — deployed entirely on a $33 ESP32-S3 microcontroller.

## 📌 Project Overview
This project targets **precision agriculture** by automating the tomato ripening process. It uses **direct pixel statistics** (RGB mean, std, mode, and Chromatic Index) for ripeness estimation and a **DQN-distilled MLP policy** to control environmental temperature. The entire pipeline is simulated and visualized via a **Digital Twin** web interface.

## 🚀 Key Features
*   **Computer Vision**: Direct pixel-based Chromatic Index extraction — no CNN, microsecond computation.
*   **Reinforcement Learning**: DQN agent distilled to a 64×64 MLP student (97.8% fidelity).
*   **Digital Twin**: Real-time visualization of ripening process and agent decisions.
*   **ESP32-S3 Deployment**: Pure-C inference on-device — 237 KB binary, no ML library needed.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Seuketchi/edge-rl-tomato-ripening.git
    cd edge-rl-tomato-ripening
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Project Structure

```
├── ml_training/           # Machine Learning Pipeline
│   ├── vision/            # (Legacy — direct pixel extraction replaced CNN)
│   ├── rl/                # Reinforcement Learning Module
│   │   ├── train_dqn.py   # Train DQN agent
│   │   ├── distill.py     # Distill teacher → student MLP
│   │   ├── export_policy_c.py  # Export weights → C headers
│   │   └── simulator.py   # Tomato ripening ODE environment
│   └── config.yaml        # Global hyperparameters
│
├── edge_firmware/         # ESP-IDF Firmware (ESP32-S3)
│   ├── main/
│   │   ├── app_main.c     # Entry point, FreeRTOS task creation
│   │   ├── edge_rl_policy.c   # Pure-C MLP forward pass
│   │   ├── edge_rl_vision.c   # Direct pixel RGB statistics extraction
│   │   ├── task_policy.c  # RL inference task + ODE simulation
│   │   ├── policy_weights.h   # Auto-generated FP32 weights
│   │   └── golden_vectors.h   # 20 test vectors for validation
│   └── CMakeLists.txt
│
├── digital_twin_viz/      # Visualization Dashboard
│   ├── server.py          # WebSocket backend
│   └── index.html         # Main dashboard interface
│
├── docs/                  # Documentation & Thesis materials
└── requirements.txt       # Python dependencies
```

## 💻 Usage

### 1. Reinforcement Learning
Train the control policy:
```bash
python ml_training/rl/train_dqn.py
```

### 2. Export & Distill
```bash
# Distill teacher → student
python ml_training/rl/distill.py

# Export student weights to C headers
PYTHONPATH=. python ml_training/rl/export_policy_c.py --verify
```

### 3. Digital Twin Demo
```bash
python digital_twin_viz/server.py
```
*Then open `digital_twin_viz/index.html` in your browser.*

### 4. ESP32 Deployment

#### Target Hardware

| Spec | Value |
|---|---|
| Board | ESP32-S3-CAM N16R8 |
| MCU | Dual-core Xtensa LX7, up to 240 MHz |
| Flash | 16 MB |
| PSRAM | 8 MB |
| Camera | OV2640 (2MP) |
| Wireless | WiFi 802.11 b/g/n, Bluetooth 5.0 LE |
| Interface | USB Type-C (programming + power) |
| Op. Temp | -20°C to +70°C |

#### Build and Flash
```bash
# 1. Build firmware
source ~/esp/v5.5.2/esp-idf/export.sh
cd edge_firmware
idf.py set-target esp32s3
idf.py build

# 2. Flash and monitor (connect board via USB-C)
idf.py -p /dev/ttyUSB0 flash monitor
```

#### Troubleshooting USB Connection
If `/dev/ttyUSB0` doesn't appear:
```bash
# Load CH9102/CH341 driver (common on ESP32-S3-CAM boards)
sudo modprobe ch341

# Add user to dialout group (one-time)
sudo usermod -aG dialout $USER

# Unplug and replug USB-C cable, then check:
ls /dev/ttyUSB*
```

#### Build Results (current)
```
Binary:  edge_rl_tomato.bin (237 KB)
Target:  ESP32-S3, ESP-IDF v5.5.2
Flash:   77% free (805 KB remaining)
Policy:  16D → 64 → 64 → 3 MLP (5,443 FP32 params)
Accuracy: 97.78% vs teacher
```

## 👥 Authors
*   **Tristan O. Jadman** - *Computer Engineering*
*   **Engr. Francis Jann Alagon** - *Adviser*
