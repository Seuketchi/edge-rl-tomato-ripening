#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

/* ============================================================
 * vision_stats.h — Shared vision statistics structure
 *
 * Used by edge_rl_vision.c (producer) and task_vision.c (consumer).
 * ============================================================ */

typedef struct {
    float mean[3];       /* RGB channel means      [0, 1] */
    float std[3];        /* RGB channel std devs    [0, 1] */
    float mode[3];       /* RGB channel modes       [0, 1] */
    float chromatic_x;   /* Chromatic Index X = G/(R+G) [0, 1] */
} vision_stats_t;

/* Implemented in edge_rl_vision.c */
esp_err_t edge_rl_vision_infer(const uint8_t *fb_buf, size_t fb_len,
                                uint8_t *out_class, float *out_confidence,
                                vision_stats_t *out_stats);
