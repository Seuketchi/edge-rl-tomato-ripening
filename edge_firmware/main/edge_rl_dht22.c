/* ============================================================
 * edge_rl_dht22.c — DHT22 temperature/humidity driver
 *
 * Bit-bangs the DHT22 single-wire protocol on the configured GPIO.
 * Timing is critical — runs with interrupts disabled during pulse
 * measurement; keep the call site tolerant of occasional failures.
 * ============================================================ */

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "rom/ets_sys.h"

#include "app_config.h"

#define TAG_DHT "DHT22"
#define DHT22_TIMEOUT_US    1000

static int64_t edge_rl_dht22_wait_level(gpio_num_t gpio, int level, int64_t timeout_us)
{
    int64_t start = esp_timer_get_time();
    while (gpio_get_level(gpio) != level) {
        if ((esp_timer_get_time() - start) > timeout_us) {
            return -1;
        }
    }
    return esp_timer_get_time() - start;
}

esp_err_t edge_rl_dht22_read(gpio_num_t gpio, float *out_temp, float *out_humidity)
{
    uint8_t data[5] = {0};

    /* Send start signal: pull low >1ms, then release */
    gpio_set_direction(gpio, GPIO_MODE_OUTPUT);
    gpio_set_level(gpio, 0);
    ets_delay_us(1200);
    gpio_set_level(gpio, 1);
    ets_delay_us(30);
    gpio_set_direction(gpio, GPIO_MODE_INPUT);

    /* Wait for sensor response: 80µs low, 80µs high */
    if (edge_rl_dht22_wait_level(gpio, 0, DHT22_TIMEOUT_US) < 0) return ESP_ERR_TIMEOUT;
    if (edge_rl_dht22_wait_level(gpio, 1, DHT22_TIMEOUT_US) < 0) return ESP_ERR_TIMEOUT;
    if (edge_rl_dht22_wait_level(gpio, 0, DHT22_TIMEOUT_US) < 0) return ESP_ERR_TIMEOUT;

    /* Read 40 bits */
    for (int i = 0; i < 40; i++) {
        if (edge_rl_dht22_wait_level(gpio, 1, DHT22_TIMEOUT_US) < 0) return ESP_ERR_TIMEOUT;
        int64_t pulse = edge_rl_dht22_wait_level(gpio, 0, DHT22_TIMEOUT_US);
        if (pulse < 0) return ESP_ERR_TIMEOUT;
        data[i / 8] <<= 1;
        if (pulse > 40) {  /* >40µs high = bit '1' */
            data[i / 8] |= 1;
        }
    }

    /* Checksum */
    uint8_t checksum = data[0] + data[1] + data[2] + data[3];
    if (checksum != data[4]) {
        ESP_LOGW(TAG_DHT, "Checksum failed: got 0x%02X expected 0x%02X", checksum, data[4]);
        return ESP_ERR_INVALID_CRC;
    }

    *out_humidity    = ((data[0] << 8) | data[1]) * 0.1f;
    float raw_temp   = (((data[2] & 0x7F) << 8) | data[3]) * 0.1f;
    if (data[2] & 0x80) raw_temp = -raw_temp;
    *out_temp        = raw_temp;

    return ESP_OK;
}
