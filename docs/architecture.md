# System Architecture

## Overview

Edge-RL follows a **two-layer architecture** (simplified from three-layer):

```
┌──────────────────────────────────────────────────────┐
│                  DEVELOPMENT HOST                     │
│  (Your laptop/desktop — training & simulation)       │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │ Digital Twin │  │ SAC Training│  │ Model Export │ │
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
│  │ Camera   │  │ (INT8, ~300KB│  │ (INT8, ~35KB)  │ │
│  └──────────┘  └──────────────┘  └───────┬────────┘ │
│                                           │          │
│  ┌──────────┐                    ┌────────▼───────┐ │
│  │ DHT22    │───────────────────→│ Action Output  │ │
│  │ Temp/Hum │                    │ harvest/wait/  │ │
│  └──────────┘                    │ adjust temp    │ │
│                                  └────────┬───────┘ │
│                                           │          │
│  ┌────────────────────────────────────────▼───────┐ │
│  │ Serial/MQTT Output (telemetry + decisions)     │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

## Edge Device: ESP32-S3

### Hardware Components

| Component | Model | Cost | Purpose |
|---|---|---|---|
| MCU | ESP32-S3-DevKitC-1 (N16R8) | ~$8 | Core processor, 16MB Flash, 8MB PSRAM |
| Camera | OV2640 (2MP) | ~$5 | RGB image capture at 320×240 |
| Sensor | DHT22 | ~$3 | Temperature + humidity |
| LED | WS2812B strip (8 LEDs) | ~$2 | Status indicators |
| Power | USB-C cable + adapter | ~$5 | 5V power supply |
| Misc | Breadboard, wires, enclosure | ~$10 | Assembly |
| **Total** | | **~$33** | |

### Memory Map (ESP32-S3 N16R8)

```
Internal SRAM (512KB):
  ├── FreeRTOS kernel + stacks    ~80KB
  ├── WiFi/MQTT stack             ~60KB
  ├── Sensor buffers              ~10KB
  ├── RL policy weights (INT8)    ~35KB
  ├── Inference scratch space     ~50KB
  └── Available                   ~277KB

External PSRAM (8MB):
  ├── Camera frame buffer         ~150KB (320×240 RGB)
  ├── Resized input buffer        ~150KB (224×224 RGB float)
  ├── Vision model weights (INT8) ~300KB
  ├── Intermediate activations    ~500KB
  └── Available                   ~6.9MB

External Flash (16MB):
  ├── Bootloader                  ~32KB
  ├── Partition table             ~4KB
  ├── Application firmware        ~2MB
  ├── Model data partition        ~2MB
  ├── OTA partition (app backup)  ~2MB
  ├── NVS (config storage)        ~64KB
  └── Available                   ~9.9MB
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
- **Algorithm:** Soft Actor-Critic (SAC) via Stable Baselines3
- **Environment:** Custom Gymnasium env wrapping digital twin simulator
- **State space:** [ripeness_stage, temperature, humidity, days_elapsed, target_harvest_day, ...]
- **Action space:** Discrete {maintain, heat, cool, harvest}
- **Training:** ~500K steps, ~8 hours on CPU
- **Distillation:** Teacher (256×256 MLP) → Student (64×64 MLP)
- **Quantization:** FP32 → INT8, final size ~35KB

### Digital Twin Simulator
Physics-based ripening model:
```
dR/dt = k₁ × (T - T_base) × (1 - R/R_max)

where:
  R = ripeness stage [0-5]
  T = temperature [12.5°C - 25°C]
  k₁ = ripening rate constant
  T_base = 12.5°C (minimum ripening temperature)
  R_max = 5.0 (fully ripe)
```

Domain randomization: temperature noise, sensor error, initial condition variation.

## Communication

- **Primary:** Serial (USB) for development and debugging
- **Optional:** MQTT over WiFi for remote monitoring
- **Telemetry format:** JSON `{"ripeness": 3, "temp": 21.5, "humidity": 68, "action": "wait", "confidence": 0.92}`
