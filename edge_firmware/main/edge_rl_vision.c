/* ============================================================
 * edge_rl_vision.c — MobileNetV2 ONNX inference (stub) + RGB Stats
 *
 * Extracts 9 RGB statistics (mean, std, mode) required by Variant B
 * and produces the Continuous Chromatic Index (X).
 * ============================================================ */

#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_err.h"
#include "esp_camera.h"
#include "app_config.h"

#define TAG_VISION "VISION"

typedef struct {
    float mean[3];
    float std[3];
    float mode[3];
    float chromatic_x;
} vision_stats_t;

esp_err_t edge_rl_vision_infer(const uint8_t *jpg_buf, size_t jpg_len,
                                uint8_t *out_class, float *out_confidence,
                                vision_stats_t *out_stats)
{
    if (jpg_buf == NULL || out_class == NULL || out_confidence == NULL || out_stats == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    /* 1. Decode JPEG to RGB888 */
    size_t rgb_len = VISION_INPUT_W * VISION_INPUT_H * 3;
    uint8_t *rgb_buf = malloc(rgb_len);
    if (!rgb_buf) {
        ESP_LOGE(TAG_VISION, "Failed to allocate RGB buffer");
        return ESP_ERR_NO_MEM;
    }

    if (!fmt2rgb888(jpg_buf, jpg_len, PIXFORMAT_JPEG, rgb_buf)) {
        ESP_LOGE(TAG_VISION, "JPEG to RGB888 decode failed");
        free(rgb_buf);
        return ESP_FAIL;
    }

    /* 2. Compute RGB Statistics */
    uint32_t sum[3] = {0};
    uint64_t sum_sq[3] = {0};
    uint32_t hist[3][16] = {0}; // 16-bin histogram for mode

    int num_pixels = VISION_INPUT_W * VISION_INPUT_H;

    for (int i = 0; i < num_pixels; i++) {
        uint8_t r = rgb_buf[i * 3 + 0];
        uint8_t g = rgb_buf[i * 3 + 1];
        uint8_t b = rgb_buf[i * 3 + 2];

        sum[0] += r; sum[1] += g; sum[2] += b;
        sum_sq[0] += (uint64_t)r * r; sum_sq[1] += (uint64_t)g * g; sum_sq[2] += (uint64_t)b * b;

        hist[0][r >> 4]++;
        hist[1][g >> 4]++;
        hist[2][b >> 4]++;
    }

    for (int c = 0; c < 3; c++) {
        // Mean [0, 1]
        out_stats->mean[c] = (float)sum[c] / (num_pixels * 255.0f);
        
        // Std [0, 1]
        float mean_val = (float)sum[c] / num_pixels;
        float variance = ((float)sum_sq[c] / num_pixels) - (mean_val * mean_val);
        out_stats->std[c] = sqrtf(fmaxf(0.0f, variance)) / 255.0f;
        
        // Mode (bin center) [0, 1]
        int max_bin = 0;
        uint32_t max_count = 0;
        for (int b = 0; b < 16; b++) {
            if (hist[c][b] > max_count) {
                max_count = hist[c][b];
                max_bin = b;
            }
        }
        out_stats->mode[c] = (max_bin * 16 + 8) / 255.0f;
    }

    // Heuristic Chromatic Index X (ROYG: 1=Green, 0=Red)
    float r_mean = out_stats->mean[0];
    float g_mean = out_stats->mean[1];
    out_stats->chromatic_x = g_mean / (r_mean + g_mean + 1e-6f);

    free(rgb_buf);

    /* STUB: ML inference */
    if (out_stats->chromatic_x > 0.55f) *out_class = CLASS_UNRIPE;
    else *out_class = CLASS_RIPE;
    *out_confidence = 0.85f;

    return ESP_OK;
}
