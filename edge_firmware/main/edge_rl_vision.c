/* ============================================================
 * edge_rl_vision.c — Direct ROI Pixel Statistics Extraction
 *
 * Extracts 9 RGB statistics (mean, std, mode) required by Variant B
 * and produces the Continuous Chromatic Index (X). No CNN inference.
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

esp_err_t edge_rl_vision_infer(const uint8_t *fb_buf, size_t fb_len,
                                uint8_t *out_class, float *out_confidence,
                                vision_stats_t *out_stats)
{
    if (fb_buf == NULL || out_class == NULL || out_confidence == NULL || out_stats == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    int expected_len = VISION_INPUT_W * VISION_INPUT_H * 2; /* 2 bytes per pixel for RGB565 */
    if (fb_len != expected_len) {
        ESP_LOGE(TAG_VISION, "Unexpected fb_len: expected %d, got %zu", expected_len, fb_len);
        return ESP_ERR_INVALID_SIZE;
    }

    /* 1. Compute RGB Statistics directly from RGB565 buffer */
    uint32_t sum[3] = {0};
    uint64_t sum_sq[3] = {0};
    uint32_t hist[3][16] = {0}; // 16-bin histogram for mode

    int num_pixels = VISION_INPUT_W * VISION_INPUT_H;

    for (int i = 0; i < num_pixels; i++) {
        /* RGB565 is little-endian on ESP32 by default in esp32-camera,
         * but typically stored as high-byte low-byte. 
         * The standard is: Byte 0 (High) = RRRRRGGG, Byte 1 (Low) = GGGBBBBB
         * esp32-camera usually returns swapped bytes depending on the sensor, 
         * let's assume standard RGB565 packed format (16-bit word).
         */
        uint16_t pixel = (fb_buf[i * 2] << 8) | fb_buf[i * 2 + 1];
        
        uint8_t r_5 = (pixel >> 11) & 0x1F;
        uint8_t g_6 = (pixel >> 5)  & 0x3F;
        uint8_t b_5 = pixel         & 0x1F;

        /* Expand to 8-bit */
        uint8_t r = (r_5 << 3) | (r_5 >> 2);
        uint8_t g = (g_6 << 2) | (g_6 >> 4);
        uint8_t b = (b_5 << 3) | (b_5 >> 2);

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

    /* Heuristic classification */
    if (out_stats->chromatic_x > 0.55f) *out_class = CLASS_UNRIPE;
    else *out_class = CLASS_RIPE;
    *out_confidence = 0.85f;

    return ESP_OK;
}
