# Edge-RL Tomato Digital Twin

An end-to-end framework for autonomous tomato ripening control, integrating Edge AI Computer Vision, Reinforcement Learning (RL), and a Digital Twin visualization system.

## 📌 Project Overview
This project targets **precision agriculture** by automating the tomato ripening process. It uses a **computer vision** model to detect ripeness stages from camera inputs and an **RL agent (SAC)** to control environmental parameters (temperature/humidity). The entire pipeline is simulated and visualized via a **Digital Twin** web interface.

## 🚀 Key Features
*   **Computer Vision**: MobileNetV2-based ripeness classifier optimized for Edge deployment (ESP-DL).
*   **Reinforcement Learning**: Soft Actor-Critic (SAC) agent trained to optimize ripening time and quality.
*   **Digital Twin**: Real-time 3D/2D visualization of the ripening process, environment state, and agent decisions.
*   **Edge Optimization**: Tools for model distillation and quantization for ESP32-S3 deployment.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/edge-rl-tomato-twin.git
    cd edge-rl-tomato-twin
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Project Structure

```
├── ml_training/           # Machine Learning Pipeline
│   ├── vision/            # Computer Vision Module
│   │   ├── train.py       # Train ripeness classifier
│   │   ├── download_dataset.py # Fetch dataset
│   │   └── export_espdl.py # Quantize for ESP32
│   ├── rl/                # Reinforcement Learning Module
│   │   ├── train_sac.py   # Train SAC agent
│   │   ├── distill.py     # Distill RL policy for edge
│   │   └── simulator.py   # Tomato ripening environment
│   └── config.yaml        # Global hyperparameters
│
├── digital_twin_viz/      # Visualization Dashboard
│   ├── src/               # Frontend assets
│   ├── server.py          # WebSocket backend
│   └── index.html         # Main dashboard interface
│
├── docs/                  # Documentation & Thesis materials
└── requirements.txt       # Python dependencies
```

## 💻 Usage

### 1. Computer Vision
Train the ripeness classifier:
```bash
# Download dataset first
python ml_training/vision/download_dataset.py

# Train the model
python ml_training/vision/train.py
```

### 2. Reinforcement Learning
Train the control policy:
```bash
python ml_training/rl/train_sac.py
```

### 3. Digital Twin Demo
Run the standalone simulation demo:
```bash
python run_sim_demo.py
```
*This generates a trajectory plot in `outputs/`.*

To run the interactive web dashboard (requires backend server):
```bash
python digital_twin_viz/server.py
```
*Then open `digital_twin_viz/index.html` in your browser.*

## 👥 Authors
*   **Tristan O. Jadman** - *Computer Engineering*
*   **Engr. Francis Jann Alagon** - *Adviser*
