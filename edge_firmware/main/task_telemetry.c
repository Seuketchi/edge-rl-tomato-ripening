/* ============================================================
 * task_telemetry.c — Serial telemetry output task
 *
 * Logs a structured JSON line to UART every TELEMETRY_INTERVAL_MS.
 * Easily parseable by a host PC for thesis data collection.
 *
 * Output format (one line per interval):
 *   {"t":25.1,"h":72.3,"class":1,"conf":0.91,"action":0,"day":2}
 * ============================================================ */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "app_config.h"
#include "shared_state.h"

static const char *CLASS_NAMES[2] = {
    "unripe", "ripe"
};

static const char *ACTION_NAMES[POLICY_NUM_ACTIONS] = {
    "maintain", "heat", "cool"
};

void edge_rl_task_telemetry(void *pvParameters)
{
    ESP_LOGI(TAG, "[telemetry] Task started on core %d", xPortGetCoreID());

    uint64_t uptime_s = 0;

    while (1) {
        uptime_s = esp_timer_get_time() / 1000000ULL;

        /* Snapshot state */
        float temp = 0, hum = 0, cls_conf = 0, act_conf = 0;
        uint8_t cls = 0, action = 0;
        uint32_t day = 0;
        bool harvest = false, vision_valid = false;

        if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            temp        = g_state.temperature_c;
            hum         = g_state.humidity_pct;
            cls         = g_state.ripeness_class;
            cls_conf    = g_state.class_confidence;
            action      = g_state.action;
            act_conf    = g_state.action_confidence;
            day         = g_state.day_counter;
            harvest     = g_state.harvest_triggered;
            vision_valid= g_state.vision_valid;
            xSemaphoreGive(g_state_mutex);
        }

        /* JSON telemetry line — parseable by Python serial reader */
        printf("{\"uptime\":%llu,\"temp\":%.1f,\"humidity\":%.1f,"
               "\"class\":\"%s\",\"class_conf\":%.2f,"
               "\"action\":\"%s\",\"action_conf\":%.3f,"
               "\"day\":%lu,\"harvest\":%s,\"vision_valid\":%s}\n",
               (unsigned long long)uptime_s,
               temp, hum,
               vision_valid ? CLASS_NAMES[cls] : "none", cls_conf,
               ACTION_NAMES[action], act_conf,
               (unsigned long)day,
               harvest      ? "true" : "false",
               vision_valid ? "true" : "false");

        vTaskDelay(pdMS_TO_TICKS(TELEMETRY_INTERVAL_MS));
    }
}
