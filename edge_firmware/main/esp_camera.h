/* esp_camera.h — Stub for compilation without esp32-camera component.
 * Real implementation requires:
 *   idf_component_register(... REQUIRES esp32-camera)
 *   or: idf.py add-dependency "espressif/esp32-camera"
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

/* Minimal type definitions to satisfy compilation */

typedef enum {
    PIXFORMAT_RGB565 = 0,
    PIXFORMAT_YUV422,
    PIXFORMAT_GRAYSCALE,
    PIXFORMAT_JPEG,
    PIXFORMAT_RGB888,
} pixformat_t;

typedef enum {
    FRAMESIZE_96X96 = 0,
    FRAMESIZE_QQVGA,
    FRAMESIZE_QCIF,
    FRAMESIZE_HQVGA,
    FRAMESIZE_240X240,
    FRAMESIZE_QVGA,
} framesize_t;

typedef enum {
    CAMERA_FB_IN_DRAM = 0,
    CAMERA_FB_IN_PSRAM,
} camera_fb_location_t;

typedef enum {
    CAMERA_GRAB_WHEN_EMPTY = 0,
    CAMERA_GRAB_LATEST,
} camera_grab_mode_t;

/* Stub LEDC defines (normally from driver) */
#ifndef LEDC_TIMER_0
#define LEDC_TIMER_0   0
#endif
#ifndef LEDC_CHANNEL_0
#define LEDC_CHANNEL_0 0
#endif

typedef struct {
    int pin_pwdn;
    int pin_reset;
    int pin_xclk;
    int pin_sscb_sda;
    int pin_sscb_scl;
    int pin_d7, pin_d6, pin_d5, pin_d4;
    int pin_d3, pin_d2, pin_d1, pin_d0;
    int pin_vsync, pin_href, pin_pclk;
    int xclk_freq_hz;
    int ledc_timer;
    int ledc_channel;
    pixformat_t pixel_format;
    framesize_t frame_size;
    int jpeg_quality;
    size_t fb_count;
    int fb_location;
    int grab_mode;
} camera_config_t;

typedef struct {
    uint8_t *buf;
    size_t   len;
    size_t   width;
    size_t   height;
    pixformat_t format;
} camera_fb_t;

/* Stub functions — return errors since no camera hardware is present */
static inline esp_err_t esp_camera_init(const camera_config_t *config) {
    (void)config;
    return ESP_ERR_NOT_SUPPORTED;
}

static inline camera_fb_t *esp_camera_fb_get(void) {
    return NULL;
}

static inline void esp_camera_fb_return(camera_fb_t *fb) {
    (void)fb;
}
