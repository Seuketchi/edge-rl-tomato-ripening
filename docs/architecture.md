# System Architecture

## Overview

Edge-RL follows a **two-layer architecture** (simplified from three-layer):

```
┌──────────────────────────────────────────────────────┐
│                  DEVELOPMENT HOST                     │
│  (Your laptop/desktop — training & simulation)       │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │ Digital Twin │  │ DQN Training│  │ Model Export │ │
│  │ Simulator    │→ │ (SB3)       │→ │ & Quantize   │ │
│  └─────────────┘  └─────────────┘  └──────┬───────┘ │
│                                            │         │
│  ┌─────────────┐  ┌─────────────┐          │         │
│  │ Kaggle Data  │→ │ Vision Model│→─────────┤         │
│  │ + Augment    │  │ Training    │          │         │
│  └─────────────┘  └─────────────┘          │         │
└────────────────────────────────────┬───────┘─────────┘
                                     │ Flash via USB
┌────────────────────────────────────▼─────────────────┐
│                   ESP32-S3 EDGE DEVICE                │
│                                                       │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ OV2640   │→ │ Vision Model │→ │ RL Policy      │ │
│  │ Camera   │  │ (INT8, ~300KB│  │ (FP32, ~21KB)  │ │
│  └──────────┘  └──────────────┘  └───────┬────────┘ │
│                                           │          │
│  ┌──────────┐                    ┌────────▼───────┐ │
│  │ DHT22    │───────────────────→│ Action Output  │ │
│  │ Temp/Hum │                    │ maintain/heat/ │ │
│  └──────────┘                    │ cool (±1°C)    │ │
│                                  └────────┬───────┘ │
│                                           │          │
│  ┌────────────────────────────────────────▼───────┐ │
│  │ Serial/MQTT Output (telemetry + decisions)     │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## Edge Device: ESP32-S3

### Hardware Components

| Component | Model | Details |
|---|---|---|
| Board | ESP32-S3-CAM N16R8 | Dual-core Xtensa LX7, up to 240 MHz |
| Flash | 16 MB | SPI Flash |
| PSRAM | 8 MB | Octal SPI PSRAM |
| Camera | OV2640 (2MP) / OV5640 (5MP) | JPEG, YUV, RGB565 formats |
| Sensor | DHT22 | Temperature + humidity |
| Wireless | WiFi 802.11 b/g/n + BLE 5.0 | 2.4 GHz |
| Interface | USB Type-C | Programming, power, debugging |
| Storage | MicroSD slot | Up to 32 GB |
| Dimensions | 54 mm × 25 mm × 13 mm | Without camera cable |
| Op. Temp | -20°C to +70°C | |

### Memory Map (ESP32-S3 N16R8)

```
Internal SRAM (512KB):
  ├── FreeRTOS kernel + stacks    ~80KB
  ├── WiFi/BLE stack               ~60KB
  ├── Sensor buffers              ~10KB
  ├── RL policy weights (FP32)     ~21KB
  ├── Inference scratch space      ~2KB
  └── Available                   ~277KB

External PSRAM (8MB):
  ├── Camera frame buffer         ~150KB (320×240 RGB)
  ├── Resized input buffer        ~150KB (224×224 RGB float)
  ├── Vision model weights (INT8) ~300KB
  ├── Intermediate activations    ~500KB
  └── Available                   ~6.9MB

External Flash (16MB):
  ├── Bootloader                  ~21KB
  ├── Partition table             ~4KB
  ├── Application firmware        ~237KB (current build)
  ├── Model data partition        ~2MB (reserved)
  ├── OTA partition (app backup)  ~2MB
  ├── NVS (config storage)        ~64KB
  └── Available                   ~11.7MB
```

### FreeRTOS Task Architecture

| Task | Priority | Core | Frequency | Purpose |
|---|---|---|---|---|
| `camera_task` | 5 | Core 1 | Every 30 min | Capture + preprocess image |
| `inference_task` | 4 | Core 1 | On new image | Run vision model + RL policy |
| `sensor_task` | 3 | Core 0 | Every 60 sec | Read DHT22 temp/humidity |
| `comms_task` | 2 | Core 0 | Every 5 min | MQTT publish telemetry |
| `watchdog_task` | 1 | Core 0 | Every 10 sec | System health monitoring |

## Development Host: ML Pipeline

### Vision Model Training
- **Dataset:** Kaggle tomato ripeness dataset (~8,000-10,000 images, 6 classes)
- **Model:** MobileNetV2-tiny or EfficientNet-Lite0 (proven on ESP32)
- **Training:** PyTorch, ~4 hours on consumer GPU
- **Quantization:** FP32 → INT8 via TensorFlow Lite or ONNX Runtime
- **Export:** C header array for direct embedding in firmware

### RL Policy Training
- **Algorithm:** Deep Q-Network (DQN) via Stable Baselines3
- **Environment:** Custom Gymnasium env wrapping digital twin simulator
- **State space (3 ablation variants):**
  - **Option A (7D):** [X, Ẋ, X_ref, T, H, t_e, t_rem]
  - **Option B (16D):** A + C_μ(3) + C_σ(3) + C_mode(3)
  - **Option C (20D):** B + max_pool(4)
- **Action space:** Discrete {maintain, heat(+ΔT), cool(−ΔT)} — 3 actions, incremental ±1°C
- **Harvest:** Automatic post-processing when X ≤ 0.15 or t_rem ≤ 0
- **Training:** ~500K steps, ~8 hours on CPU
- **Distillation:** Teacher (256×256 MLP) → Student (64×64 MLP), 97.78% accuracy
- **Deployment:** FP32, ~21KB weights, pure C inference (no ML library)

### Digital Twin Simulator
Physics-based chromatic evolution ODE (ROYG convention):
```
dX/dt = −k₁ × (T - T_base) × X

where:
  X = Continuous Chromatic Index [0-1]
      ROYG convention: X=1.0 (Green/unripe) → X=0.0 (Red/ripe)
  T = temperature [12.5°C - 35°C]
  k₁ = cultivar-specific ripening rate constant
  T_base = 12.5°C (minimum ripening temperature)

Analytical reference trajectory:
  X_ref(t) = exp(−k₁ × (T_ideal − T_base) × t)
```

Domain randomization: temperature noise, sensor error, initial condition variation.

## Communication

- **Primary:** Serial (USB) for development and debugging
- **Optional:** MQTT over WiFi for remote monitoring
- **Telemetry format:** JSON `{"X": 0.72, "dX_dt": -0.03, "temp": 21.5, "humidity": 68, "t_rem": 4.5, "action": "maintain", "harvest_ready": false}`
