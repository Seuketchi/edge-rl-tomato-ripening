/* ============================================================
 * task_vision.c — Vision inference task 
 *
 * Waits for a frame from g_camera_queue, computes the direct
 * pixel RGB statistics and Chromatic Index via edge_rl_vision_infer.
 *
 * NOTE: The implementation is in edge_rl_vision.c.
 * ============================================================ */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "driver/gpio.h"

#include "app_config.h"
#include "shared_state.h"

typedef struct {
    float mean[3];
    float std[3];
    float mode[3];
    float chromatic_x;
} vision_stats_t;

/* From components/edge_rl_vision */
esp_err_t edge_rl_vision_infer(const uint8_t *jpg_buf, size_t jpg_len,
                                uint8_t *out_class, float *out_confidence,
                                vision_stats_t *out_stats);

/* Queue declared in task_camera.c */
extern QueueHandle_t g_camera_queue;

void edge_rl_task_vision(void *pvParameters)
{
    ESP_LOGI(TAG, "[vision] Task started on core %d", xPortGetCoreID());

    camera_fb_t *fb = NULL;

    while (1) {
        /* Block until camera delivers a frame (max 35-min wait) */
        if (xQueueReceive(g_camera_queue, &fb, pdMS_TO_TICKS(35 * 60 * 1000)) != pdTRUE) {
            ESP_LOGW(TAG, "[vision] No frame received — waiting");
            continue;
        }

        int64_t t_start = esp_timer_get_time();

        uint8_t cls = 0;
        float   conf = 0.0f;
        vision_stats_t stats = {0};
        
        gpio_set_level(PROFILING_PIN_ML, 1); /* START PROFILING ML ENERGY */
        esp_err_t err = edge_rl_vision_infer(fb->buf, fb->len, &cls, &conf, &stats);
        gpio_set_level(PROFILING_PIN_ML, 0); /* STOP PROFILING ML ENERGY  */

        int64_t latency_ms = (esp_timer_get_time() - t_start) / 1000;
        esp_camera_fb_return(fb);

        if (err != ESP_OK) {
            ESP_LOGW(TAG, "[vision] Inference failed: %s", esp_err_to_name(err));
            continue;
        }

        ESP_LOGI(TAG, "[vision] X=%.3f class=%d conf=%.2f latency=%lldms", 
                 stats.chromatic_x, cls, conf, latency_ms);

        if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            g_state.prev_chromatic_x = g_state.chromatic_x;
            g_state.chromatic_x      = stats.chromatic_x;
            
            for(int i=0; i<3; i++) {
                g_state.rgb_mean[i] = stats.mean[i];
                g_state.rgb_std[i]  = stats.std[i];
                g_state.rgb_mode[i] = stats.mode[i];
            }
            
            g_state.ripeness_class   = cls;
            g_state.class_confidence = conf;
            g_state.vision_valid     = true;
            xSemaphoreGive(g_state_mutex);
        }
    }
}
