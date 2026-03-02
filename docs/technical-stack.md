# Technical Stack Reference

## Hardware

| Component | Specification | Notes |
|---|---|---|
| MCU | ESP32-S3-DevKitC-1 (N16R8) | 240MHz dual-core, 512KB SRAM, 8MB PSRAM, 16MB Flash |
| Camera | OV2640 | 2MP, JPEG/RGB565/YUV, max 1600×1200; we use 320×240 |
| Temp/Humidity | DHT22 (AM2302) | ±0.5°C, ±2% RH, 0.5Hz sampling |
| Status LEDs | WS2812B (NeoPixel) | 8 addressable RGB LEDs |
| Power | USB-C 5V/500mA | ~1.5W peak consumption |

## Edge Firmware

| Layer | Technology | Version |
|---|---|---|
| Framework | ESP-IDF | v5.1+ |
| RTOS | FreeRTOS | (bundled with ESP-IDF) |
| ML Runtime | Pure C MLP (no runtime) | — |
| Camera Driver | esp32-camera | v2.0+ |
| Networking | esp-mqtt + esp-wifi | (bundled with ESP-IDF) |
| Language | C (C11) | — |

### Key ESP-IDF Components
- `esp_camera` — OV2640 driver with DMA
- `esp_mqtt` — lightweight MQTT client
- `nvs_flash` — non-volatile storage for config
- `esp_ota` — over-the-air firmware updates (optional)

## ML Training (Development Host)

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Primary language |
| PyTorch | 2.1+ | Vision model training |
| torchvision | 0.16+ | Pretrained models, transforms |
| Stable Baselines3 | 2.1+ | DQN RL training |
| Gymnasium | 0.29+ | RL environment interface |
| ONNX / ONNX Runtime | 1.15+ | Optional model export path |
| NumPy | 1.24+ | Numerical operations |
| Matplotlib / Seaborn | Latest | Visualization for thesis figures |
| scikit-learn | 1.3+ | Metrics (confusion matrix, etc.) |
| Weights & Biases | Latest | Experiment tracking (optional) |
| pandas | 2.1+ | Data analysis |

### requirements.txt
```
torch>=2.1.0
torchvision>=0.16.0
stable-baselines3>=2.1.0
gymnasium>=0.29.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.66.0
```

## Vision Pipeline

| Approach | Size | Notes |
|---|---|---|
| Direct Pixel Statistics | ~1KB code | RGB mean/std/mode + Chromatic Index X = G/(R+G) |

> Direct pixel extraction guarantees identical input distributions between simulator and hardware, runs in microseconds, and needs no ML runtime.

## RL Configuration

```yaml
algorithm: DQN
policy: MlpPolicy
state_dim: 16    # Variant B: X, dX/dt, X_ref, C_mu(3), C_sig(3), C_mode(3), T, H, t_e, t_rem
action_space: Discrete(3)  # maintain, heat, cool
learning_rate: 3e-4
buffer_size: 100_000
batch_size: 256
gamma: 0.99
tau: 0.005
total_timesteps: 500_000
```

### Policy Distillation
- Teacher: `[16] → [256] → [256] → [3]` (DQN, ~500KB FP32)
- Student: `[16] → [64] → [64] → [3]` (5,443 params, ~21.8KB FP32)
- Distillation loss: Cross-entropy (action imitation)
- Exported as FP32 C header for pure-C inference

## Validation Dataset

### Real-World Validation
- 5-10 real tomato trials captured with OV2640 at 96×96 RGB565
- Ground-truth via manual colour assessment
- Used for sim-to-real transfer evaluation (RQ3)

## Monitoring & Logging

- **Development:** Serial monitor at 115200 baud (`idf.py monitor`)
- **Telemetry:** MQTT to local broker (Mosquitto) or cloud (optional)
- **Data format:** JSON over MQTT topic `edge-rl/telemetry`
- **Logging:** ESP-IDF `ESP_LOG*` macros with configurable verbosity
