#pragma once

#include <stdint.h>
#include <stdbool.h>

/* ============================================================
 * shared_state.h — Global state shared across FreeRTOS tasks
 *
 * Always access via g_state_mutex (xSemaphoreTake/Give).
 * ============================================================ */

typedef struct {
    /* --- Sensor readings (from task_sensors) --- */
    float temperature_c;       /* DHT22 temperature, °C */
    float humidity_pct;        /* DHT22 relative humidity, % */
    uint32_t sensor_timestamp; /* esp_timer_get_time() at last reading, ms */

    /* --- Vision output (from task_vision) --- */
    float   chromatic_x;       /* Continuous Chromatic Index X ∈ [0, 1] (ROYG: 1=Green, 0=Red) */
    float   prev_chromatic_x;  /* Previous X for dX/dt velocity calc */
    float   class_confidence;  /* softmax probability of top class */
    bool    vision_valid;      /* false until first successful inference */

    /* --- Policy output (from task_policy) --- */
    uint8_t action;            /* 0=maintain 1=heat(+ΔT) 2=cool(-ΔT) */
    float   action_confidence; /* max logit value (unnormalised) */
    uint32_t day_counter;      /* days since system start */

    /* --- System flags --- */
    bool harvest_ready;        /* set when X <= ripe threshold (post-processing) */
    bool camera_ready;         /* set after camera init succeeds */
    bool thermal_override;     /* set when hard-coded thermal guardrail fires */
} edge_rl_state_t;

/* Declared in app_main.c */
extern edge_rl_state_t g_state;
extern SemaphoreHandle_t g_state_mutex;
