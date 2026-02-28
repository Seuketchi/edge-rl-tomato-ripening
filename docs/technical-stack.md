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
| ML Runtime | ESP-DL v3.x | v3.2.0+ |
| Camera Driver | esp32-camera | v2.0+ |
| Networking | esp-mqtt + esp-wifi | (bundled with ESP-IDF) |
| Language | C (C11) | — |

### Key ESP-IDF Components
- `esp_nn` — optimized neural network kernels for ESP32-S3
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
| Stable Baselines3 | 2.1+ | SAC RL training |
| Gymnasium | 0.29+ | RL environment interface |
| TensorFlow Lite | 2.14+ | Model quantization & conversion |
| ONNX / ONNX Runtime | 1.15+ | Optional model export path |
| OpenCV | 4.8+ | Image processing |
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
opencv-python>=4.8.0
Pillow>=10.0.0
tensorflow>=2.14.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.66.0
```

## Vision Pipeline (Ranked by ESP32 Feasibility)

| Approach | Size | Accuracy (est.) | ESP32-S3 Support | Recommendation |
|---|---|---|---|---|
| Direct Pixel Statistics | ~1KB | Exact Math | ✅ Proven | **Primary choice** |
| CNN (MobileNetV2) | ~200KB | ~85% | ✅ Proven | Deprecated (covariate shift) |

> **Recommendation:** Use Direct Pixel RGB extraction. It removes the need for an ML runtime, avoids INT8 quantization loss, and perfectly matches the simulator.

## RL Configuration

```yaml
algorithm: SAC
policy: MlpPolicy
state_dim: 9
action_space: Discrete(4)  # maintain, heat, cool, harvest
learning_rate: 3e-4
buffer_size: 100_000
batch_size: 256
gamma: 0.99
tau: 0.005
total_timesteps: 500_000
```

### Policy Distillation
- Teacher: `[9] → [256] → [256] → [4]` (~500KB FP32)
- Student: `[9] → [64] → [64] → [4]` (~40KB FP32)
- Distillation loss: KL divergence + action MSE
- Post-quantization: ~35KB INT8

## Dataset

### Kaggle Tomato Ripeness (Primary)
- ~8,000-10,000 labeled images
- 6 classes: Green, Breaker, Turning, Pink, Light Red, Red
- Split: 70% train / 15% val / 15% test
- Source: [Kaggle Tomato Ripeness datasets](https://www.kaggle.com/datasets)

### Grocery-Store Validation (Real-World)
- 60 images minimum (10 per ripeness stage)
- Captured with OV2640 camera at 320×240
- Manual labeling by researcher
- Used for few-shot fine-tuning and final accuracy evaluation

## Monitoring & Logging

- **Development:** Serial monitor at 115200 baud (`idf.py monitor`)
- **Telemetry:** MQTT to local broker (Mosquitto) or cloud (optional)
- **Data format:** JSON over MQTT topic `edge-rl/telemetry`
- **Logging:** ESP-IDF `ESP_LOG*` macros with configurable verbosity
