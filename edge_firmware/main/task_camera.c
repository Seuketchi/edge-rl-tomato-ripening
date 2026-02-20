/* ============================================================
 * task_camera.c — OV2640 camera capture task
 *
 * Captures a JPEG frame every CAPTURE_INTERVAL_MS and places
 * the raw buffer into a FreeRTOS queue for task_vision to pick up.
 * ============================================================ */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_camera.h"

#include "app_config.h"
#include "shared_state.h"

/* Camera frame queue — holds one pending frame at a time */
QueueHandle_t g_camera_queue = NULL;

static esp_err_t edge_rl_camera_init(void)
{
    camera_config_t config = {
        .pin_pwdn   = CAM_PIN_PWDN,
        .pin_reset  = CAM_PIN_RESET,
        .pin_xclk   = CAM_PIN_XCLK,
        .pin_sscb_sda = CAM_PIN_SIOD,
        .pin_sscb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7, .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5, .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3, .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1, .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href  = CAM_PIN_HREF,
        .pin_pclk  = CAM_PIN_PCLK,
        .xclk_freq_hz = CAM_XCLK_FREQ_HZ,
        .ledc_timer   = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size   = CAM_FRAME_SIZE,
        .jpeg_quality = 12,
        .fb_count     = 2,
        .fb_location  = CAMERA_FB_IN_PSRAM,
        .grab_mode    = CAMERA_GRAB_WHEN_EMPTY,
    };
    return esp_camera_init(&config);
}

void edge_rl_task_camera(void *pvParameters)
{
    ESP_LOGI(TAG, "[camera] Task started on core %d", xPortGetCoreID());

    g_camera_queue = xQueueCreate(1, sizeof(camera_fb_t *));

    esp_err_t err = edge_rl_camera_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "[camera] Init failed: %s — task halted", esp_err_to_name(err));
        vTaskDelete(NULL);
        return;
    }

    if (xSemaphoreTake(g_state_mutex, portMAX_DELAY) == pdTRUE) {
        g_state.camera_ready = true;
        xSemaphoreGive(g_state_mutex);
    }
    ESP_LOGI(TAG, "[camera] OV2640 initialised");

    while (1) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb == NULL) {
            ESP_LOGW(TAG, "[camera] Frame buffer NULL — skipping");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        ESP_LOGI(TAG, "[camera] Captured frame: %zu bytes", fb->len);

        /* Drop old frame if vision hasn't consumed it yet, then enqueue */
        camera_fb_t *old = NULL;
        if (xQueueReceive(g_camera_queue, &old, 0) == pdTRUE) {
            esp_camera_fb_return(old);
        }
        xQueueSend(g_camera_queue, &fb, 0);

        vTaskDelay(pdMS_TO_TICKS(CAPTURE_INTERVAL_MS));
    }
}
