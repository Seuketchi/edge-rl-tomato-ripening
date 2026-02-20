/* ============================================================
 * app_main.c — Edge-RL: Application entry point
 *
 * Initialises hardware and spawns all FreeRTOS tasks.
 * Each task runs independently; shared state is passed via
 * edge_rl_shared_state (see shared_state.h).
 * ============================================================ */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"

#include "app_config.h"
#include "shared_state.h"

/* ---- Task entry-point declarations ---- */
void edge_rl_task_camera(void *pvParameters);
void edge_rl_task_sensors(void *pvParameters);
void edge_rl_task_vision(void *pvParameters);
void edge_rl_task_policy(void *pvParameters);
void edge_rl_task_telemetry(void *pvParameters);

/* ---- Global shared state ---- */
edge_rl_state_t g_state = {0};
SemaphoreHandle_t g_state_mutex = NULL;

void app_main(void)
{
    ESP_LOGI(TAG, "=== Edge-RL Tomato Ripening Controller ===");
    ESP_LOGI(TAG, "ESP32-S3 | ESP-IDF v5.1+");

    /* NVS required by Wi-Fi/BT stacks even if unused */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }

    /* Shared state mutex */
    g_state_mutex = xSemaphoreCreateMutex();
    ESP_ERROR_CHECK(g_state_mutex == NULL ? ESP_ERR_NO_MEM : ESP_OK);

    /* Spawn tasks */
    xTaskCreate(edge_rl_task_sensors,   "sensors",   TASK_STACK_SENSORS,   NULL, TASK_PRIORITY_SENSORS,   NULL);
    xTaskCreate(edge_rl_task_camera,    "camera",    TASK_STACK_CAMERA,    NULL, TASK_PRIORITY_CAMERA,    NULL);
    xTaskCreate(edge_rl_task_vision,    "vision",    TASK_STACK_VISION,    NULL, TASK_PRIORITY_VISION,    NULL);
    xTaskCreate(edge_rl_task_policy,    "policy",    TASK_STACK_POLICY,    NULL, TASK_PRIORITY_POLICY,    NULL);
    xTaskCreate(edge_rl_task_telemetry, "telemetry", TASK_STACK_TELEMETRY, NULL, TASK_PRIORITY_TELEMETRY, NULL);

    ESP_LOGI(TAG, "All tasks spawned. System running.");
}
