# Edge-RL Tomato Ripening — To Do

## Hardware Integration
- [ ] Flash ESP32-S3 and verify golden test + ODE simulation on serial monitor
- [ ] Wire DHT22 sensor (GPIO 32) — verify temperature/humidity readings
- [ ] Wire OV2640 camera (PSRAM required) — verify frame capture
- [ ] Wire relay module for heater control (GPIO TBD)

## Vision Model On-Device
- [ ] Integrate esp-dl or implement pure-C MobileNetV2 inference
- [ ] Replace vision stub (`edge_rl_vision.c`) with real model forward pass
- [ ] Test with real tomato images — classify ripeness stages on-device

## System Integration
- [ ] End-to-end loop: camera → vision → policy → relay (every 30 min)
- [ ] Telemetry logging: collect JSON data over serial for thesis results
- [ ] Safety testing: verify thermal guardrails cut heater at 35°C

## Thesis
- [ ] Capture serial output from golden test + ODE sim for thesis figures
- [ ] Add deployment results section (inference latency, memory, flash usage)
- [ ] Final review of all chapters and bibliography
- [ ] Submit

## Nice-to-Have
- [ ] WiFi telemetry (MQTT or HTTP POST to dashboard)
- [ ] HIL serial loop (PC simulates environment, ESP32 runs policy)
- [ ] INT8 quantization of policy weights (21 KB → 5 KB)
