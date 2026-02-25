# GitHub Repositories for Edge-RL Tomato Ripening Research

Curated list of open-source repositories directly relevant to the thesis.

---

## 🌱 RL in Agriculture & Crop Management

| Repository | Description | Why It's Useful |
|---|---|---|
| [farm-gym/farm-gym](https://github.com/farm-gym/farm-gym) | Farming Environment Gym factory for RL — build custom ag environments with Weather, Soil, Plant entities | Reference architecture for your tomato ripening Gym env |
| [poomstas/AgTech_DRL](https://github.com/poomstas/AgTech_DRL) | Optimizing plant growth with DDPG, TD3, SAC using PCSE crop simulator | Comparison baselines for DRL in crop optimization |
| [Kaushal1011/AutonomousGreenhouseControl](https://github.com/Kaushal1011/AutonomousGreenhouseControl) | RL for autonomous greenhouse control with tomato crops | **Directly relevant** — RL + tomato + greenhouse |
| [ysaikai/RLIR](https://github.com/ysaikai/RLIR) | Deep RL for irrigation scheduling with sensor feedback | DQN-based ag decision-making reference |
| [FayElhassan/AI-for-Sustainable-Agriculture](https://github.com/FayElhassan/AI-for-Sustainable-Agriculture) | PPO, A2C, SAC for greenhouse optimization | Algorithm comparison benchmarks |
| [rgreenblatt/CropGym](https://github.com/rgreenblatt/CropGym) | RL environment for crop management (from your ref [8]) | Already cited — study the env design |
| [Gym-DSSAT](https://github.com/rgautron/gym-DSSAT) | Gym wrapper for the DSSAT crop model | Professional-grade crop sim + RL interface |

---

## 🍅 Tomato Ripeness Detection & Datasets

| Repository | Description | Why It's Useful |
|---|---|---|
| [MiguelAngel-GM/tomatOD](https://github.com/MiguelAngel-GM/tomatOD) | Tomato detection dataset — 277 images, 2418 annotated fruits (unripe/semi-ripe/ripe) | **Pre-train your vision model** or use for domain randomization |
| [laboro-ai/LaboroTomato](https://github.com/laboro-ai/LaboroTomato) | Large-scale tomato dataset for detection + instance segmentation (ripe/half-ripe/green) | Multi-stage ripeness labels for classifier training |
| [Ishikajaiswal/TomatoDetection](https://github.com/Ishikajaiswal/TomatoDetection) | ResNet50-based ripe/raw tomato classifier | Reference CNN architecture for comparison |

---

## 🧠 TinyML & Edge AI on ESP32

| Repository | Description | Why It's Useful |
|---|---|---|
| [espressif/esp-dl](https://github.com/espressif/esp-dl) | Official Espressif deep learning library for ESP32-S3 | **Your deployment target** — already cited |
| [espressif/esp-tflite-micro](https://github.com/espressif/esp-tflite-micro) | TensorFlow Lite Micro for ESP32 | Alternative inference engine for ESP32 |
| [tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro) | Official TF Lite Micro — the core inference engine for MCUs | Upstream of your deployment stack |
| [mit-han-lab/mcunet](https://github.com/mit-han-lab/mcunet) | MCUNet: Tiny deep learning on IoT devices (from your ref [14]) | **Already cited** — study NAS for MCU models |
| [mlcommons/tiny](https://github.com/mlcommons/tiny) | MLPerf Tiny benchmark suite (from your ref [13]) | Benchmark your model against industry standards |
| [Aizhee/arduino-bitneural32](https://github.com/Aizhee/arduino-bitneural32) | Bitnet-inspired 1.58-bit quantized NN library for ESP32 | Extreme quantization for your DQN policy |
| [DanielStoelzner/ESP32-CAM-FreeRTOS](https://github.com/DanielStoelzner/ESP32-CAM-FreeRTOS) | ESP32-CAM streaming server built on FreeRTOS | FreeRTOS task architecture reference for camera + ML pipeline |
| [ali-aljufairi/Embedded-Project](https://github.com/ali-aljufairi/Embedded-Project) | Object detection on ESP32-CAM with TFLite + Edge Impulse + FreeRTOS | End-to-end edge ML pipeline reference |
| [maarten-pennings/esp32cam](https://github.com/maarten-pennings/esp32cam) | TFLite model deployment on ESP32-CAM from SD card | Flexible model loading approach |

---

## 🔄 Sim-to-Real Transfer & Domain Randomization

| Repository | Description | Why It's Useful |
|---|---|---|
| [montrealrobotics/domain-randomizer](https://github.com/montrealrobotics/domain-randomizer) | Standalone library to randomize OpenAI Gym environments | **Plug into your tomato sim** for domain randomization |
| [montrealrobotics/active-domainrand](https://github.com/montrealrobotics/active-domainrand) | Active Domain Randomization — searches for hardest MDP instances | Smarter DR than uniform sampling |
| [gabrieletiboni/dropo](https://github.com/gabrieletiboni/dropo) | DROPO: Offline Domain Randomization with optimal range estimation | Avoids manual DR tuning |
| [smkdGab/Sim-to-Real-transfer-of-RL-Policies-in-Robotics](https://github.com/smkdGab/Sim-to-Real-transfer-of-Reinforcement-Learning-Policies-in-Robotics) | Tools & algorithms for sim-to-real RL | Comprehensive reference codebase |
| [AwesomeSim2Real](https://github.com/SeungHyunKim1/AwesomeSim2Real) | Curated list of sim-to-real papers & code | **Literature survey accelerator** |

---

## 🎓 Policy & Knowledge Distillation

| Repository | Description | Why It's Useful |
|---|---|---|
| [CUN-bjy/policy-distillation-baselines](https://github.com/CUN-bjy/policy-distillation-baselines) | PyTorch policy distillation with Stable-Baselines3 teachers → small students | **Directly applicable** to your DQN distillation pipeline |
| [jetsnguns/realtime-policy-distillation](https://github.com/jetsnguns/realtime-policy-distillation) | Real-time Policy Distillation using Q-loss + KL loss on Ray APEX DQN | DQN-specific distillation implementation |
| [dkgupta90/awesome-knowledge-distillation](https://github.com/dkgupta90/awesome-knowledge-distillation) | Comprehensive list including RL distillation papers | Find more distillation techniques |

---

## 📦 Model Compression & Quantization

| Repository | Description | Why It's Useful |
|---|---|---|
| [tensorflow/model-optimization](https://github.com/tensorflow/model-optimization) | Official TF Model Optimization Toolkit (pruning, quantization, clustering) | **INT8 quantization** for your policy network |
| [mit-han-lab/once-for-all](https://github.com/mit-han-lab/once-for-all) | Once-for-All: Train one network, specialize for diverse hardware | NAS approach to find optimal MCU architecture |
| [htqin/awesome-model-quantization](https://github.com/htqin/awesome-model-quantization) | Curated list of quantization papers & code | Stay current on compression techniques |

---

## 🌿 Plant Growth Simulation & Digital Twin

| Repository | Description | Why It's Useful |
|---|---|---|
| [ebimodeling/biocro](https://github.com/ebimodeling/biocro) | BioCro II — modular crop growth simulation with ODE solvers | ODE-based crop models for your tomato sim |
| [WUR-AI/DeepCGM](https://github.com/WUR-AI/DeepCGM) | Deep learning crop growth model with physics constraints | Knowledge-guided simulation approach |

---

## 🛠️ Core RL Libraries (Already in Your Stack)

| Repository | Description |
|---|---|
| [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | Your RL training framework (ref [27]) |
| [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | The env interface standard |
| [ray-project/ray](https://github.com/ray-project/ray) | Scalable RL with RLlib (useful for scaling up training) |

---

---

## 📊 TensorBoard & RL Monitoring

| Repository | Description | Why It's Useful |
|---|---|---|
| [DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) | Training framework + hyperparameter tuning for SB3 with built-in TensorBoard | **Best reference** for SB3 + TensorBoard integration patterns |
| [openai/baselines](https://github.com/openai/baselines) | Original OpenAI baselines with TensorBoard logging | Classic RL monitoring patterns |
| [wandb/examples](https://github.com/wandb/examples) | Weights & Biases integration examples including SB3 | Alternative to TensorBoard with richer experiment tracking |
| [tensorboard/tensorboard](https://github.com/tensorflow/tensorboard) | Official TensorBoard repo | The visualization tool itself |

> **SB3 TensorBoard usage** (already integrated in your `train_dqn.py`):
> ```python
> model = DQN("MlpPolicy", env, tensorboard_log="./tb_logs/")
> # Then view with: tensorboard --logdir ./tb_logs/
> ```
> SB3 automatically logs: episode reward, episode length, loss, learning rate, exploration rate.

---

## ⭐ Top 5 Most Impactful for Your Thesis

1. **[policy-distillation-baselines](https://github.com/CUN-bjy/policy-distillation-baselines)** — Direct reference for distilling your DQN to a tiny network
2. **[domain-randomizer](https://github.com/montrealrobotics/domain-randomizer)** — Plug-and-play DR for your Gym environment
3. **[tomatOD](https://github.com/MiguelAngel-GM/tomatOD)** — Annotated tomato dataset for your vision pipeline
4. **[AutonomousGreenhouseControl](https://github.com/Kaushal1011/AutonomousGreenhouseControl)** — Closest existing work: RL + tomato + greenhouse
5. **[esp-tflite-micro](https://github.com/espressif/esp-tflite-micro)** — Your ESP32-S3 inference engine
