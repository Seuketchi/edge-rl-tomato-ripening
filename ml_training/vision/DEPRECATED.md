# DEPRECATED

The Convolutional Neural Network (CNN) vision pipeline previously located in this module has been **deprecated and removed from the active system architecture**.

**Reasoning:**
The system uses Direct Pixel Feature Extraction to convert raw RGB arrays into a 10-dimensional feature vector containing RGB statistics and a Chromatic Index (X). The Digital Twin simulator was calibrated and trained directly on these mathematical properties. Retaining a CNN (MobileNetV2) for edge deployment would introduce covariate shift and approximation errors without providing any additional biological context.

The direct pixel extraction guarantees identical input distributions for the reinforcement learning policy during both training and inference, runs in microseconds on the ESP32-S3, and frees up hundreds of kilobytes of flash memory.

**Status:**
These scripts are retained exclusively for historical reference and reproducing ablation studies. Do NOT attempt to integrate these models into the active ESP-IDF firmware.
