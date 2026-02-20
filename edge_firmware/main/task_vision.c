/* ============================================================
 * task_vision.c — Vision inference task (stub)
 *
 * Waits for a frame from g_camera_queue, runs the INT8
 * MobileNetV2 classifier via ESP-DL, writes result to g_state.
 *
 * NOTE: The ESP-DL model call (edge_rl_vision_infer) is
 * implemented in components/edge_rl_vision once model_data.h
 * is generated from the export pipeline (Step 4).
 * ============================================================ */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_camera.h"
#include "esp_timer.h"

#include "app_config.h"
#include "shared_state.h"

/* From components/edge_rl_vision */
esp_err_t edge_rl_vision_infer(const uint8_t *rgb_buf, size_t buf_len,
                                uint8_t *out_class, float *out_confidence);

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
        esp_err_t err = edge_rl_vision_infer(fb->buf, fb->len, &cls, &conf);

        int64_t latency_ms = (esp_timer_get_time() - t_start) / 1000;
        esp_camera_fb_return(fb);

        if (err != ESP_OK) {
            ESP_LOGW(TAG, "[vision] Inference failed: %s", esp_err_to_name(err));
            continue;
        }

        ESP_LOGI(TAG, "[vision] class=%d conf=%.2f latency=%lldms", cls, conf, latency_ms);

        if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            g_state.ripeness_class   = cls;
            g_state.class_confidence = conf;
            g_state.vision_valid     = true;
            xSemaphoreGive(g_state_mutex);
        }
    }
}
