/* ============================================================
 * edge_rl_policy.c — Distilled DQN student MLP (stub)
 *
 * policy_data.h is generated: ml_training/bin_to_c_array.py
 * Source ONNX: outputs/distill_20260213_092014/rl_policy.onnx
 *   Size: 8.9 KB  |  5,060 params  |  Input: [1,9]  Output: [1,4]
 *   Accuracy vs teacher: 98.2%
 *
 * TODO: Replace stub with real ESP-DL inference:
 *   1. #include "policy_data.h"
 *   2. Pack 9D state_vec into model input tensor
 *   3. dl::Model::run() with rl_policy_data / rl_policy_size
 *   4. Return argmax action + softmax max confidence
 * ============================================================ */

#include <string.h>
#include "esp_log.h"
#include "esp_err.h"
#include "app_config.h"

#define TAG_POLICY "POLICY"

esp_err_t edge_rl_policy_infer(const float *state, int state_dim,
                                uint8_t *out_action, float *out_confidence)
{
    if (state == NULL || out_action == NULL || out_confidence == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (state_dim != POLICY_STATE_DIM) {
        ESP_LOGE(TAG_POLICY, "State dim mismatch: expected %d, got %d",
                 POLICY_STATE_DIM, state_dim);
        return ESP_ERR_INVALID_SIZE;
    }

    /* STUB: simple heuristic mirroring fixed_stage5 baseline */
    /* Replace this body with ESP-DL model inference in Step 4 */
    ESP_LOGW(TAG_POLICY, "STUB inference — policy_data.h not yet embedded");

    float ripeness_norm = state[2];  /* normalised ripeness [0,1] */
    if (ripeness_norm >= 0.8f) {
        *out_action = ACTION_HARVEST;
    } else if (ripeness_norm >= 0.5f) {
        *out_action = ACTION_HEAT;
    } else {
        *out_action = ACTION_MAINTAIN;
    }
    *out_confidence = 0.70f;

    return ESP_OK;
}
