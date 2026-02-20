/* ============================================================
 * task_sensors.c — DHT22 sensor reading task
 *
 * Reads temperature and humidity every SENSOR_INTERVAL_MS,
 * updates g_state under mutex.
 * ============================================================ */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"

#include "app_config.h"
#include "shared_state.h"

/* Forward declaration for DHT22 driver (component: edge_rl_sensors) */
esp_err_t edge_rl_dht22_read(gpio_num_t gpio, float *out_temp, float *out_humidity);

void edge_rl_task_sensors(void *pvParameters)
{
    ESP_LOGI(TAG, "[sensors] Task started on core %d", xPortGetCoreID());

    float temp = 0.0f, humidity = 0.0f;

    while (1) {
        esp_err_t err = edge_rl_dht22_read(DHT22_GPIO, &temp, &humidity);

        if (err == ESP_OK) {
            if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                g_state.temperature_c    = temp;
                g_state.humidity_pct     = humidity;
                g_state.sensor_timestamp = (uint32_t)(esp_timer_get_time() / 1000);
                xSemaphoreGive(g_state_mutex);
            }
            ESP_LOGI(TAG, "[sensors] T=%.1f°C  H=%.1f%%", temp, humidity);
        } else {
            ESP_LOGW(TAG, "[sensors] DHT22 read failed: %s", esp_err_to_name(err));
        }

        vTaskDelay(pdMS_TO_TICKS(SENSOR_INTERVAL_MS));
    }
}
