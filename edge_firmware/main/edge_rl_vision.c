/* ============================================================
 * edge_rl_vision.c — MobileNetV2 ONNX inference (stub)
 *
 * model_data.h is generated: ml_training/bin_to_c_array.py
 * Source ONNX: outputs/vision_20260212_090651/tomato_classifier.onnx
 *   Size: 265.9 KB  |  4 classes  |  Input: 1×3×224×224
 *
 * TODO: Replace stub with real ESP-DL inference:
 *   1. #include "model_data.h"
 *   2. Decode JPEG → RGB888 via esp_jpg_decode
 *   3. Resize to VISION_INPUT_W × VISION_INPUT_H
 *   4. dl::Model::run() with vision_model_data / vision_model_size
 *   5. Apply softmax → return argmax class + confidence
 * ============================================================ */

#include <string.h>
#include "esp_log.h"
#include "esp_err.h"
#include "app_config.h"

#define TAG_VISION "VISION"

esp_err_t edge_rl_vision_infer(const uint8_t *rgb_buf, size_t buf_len,
                                uint8_t *out_class, float *out_confidence)
{
    if (rgb_buf == NULL || out_class == NULL || out_confidence == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    /* STUB: returns CLASS_RIPE with 0.85 confidence */
    /* Replace this body with ESP-DL model inference in Step 4  */
    ESP_LOGW(TAG_VISION, "STUB inference — model_data.h not yet embedded");
    *out_class      = CLASS_RIPE;
    *out_confidence = 0.85f;

    return ESP_OK;
}
