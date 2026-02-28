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
    float   rgb_mean[3];       /* RGB channel means */
    float   rgb_std[3];        /* RGB channel standard deviations */
    float   rgb_mode[3];       /* RGB channel modes */
    float   class_confidence;  /* confidence heuristic (hardcoded value) */
    bool    vision_valid;      /* false until first successful inference */

    /* --- Policy output (from task_policy) --- */
    uint8_t action;            /* 0=maintain 1=heat(+ΔT) 2=cool(-ΔT) */
    float   action_confidence; /* max logit value (unnormalised) */
    uint32_t day_counter;      /* days since system start */
    uint8_t ripeness_class;    /* vision classifier output class index */

    /* --- System flags --- */
    bool harvest_ready;        /* set when X <= ripe threshold (post-processing) */
    bool harvest_triggered;    /* set when harvest action has been sent */
    bool camera_ready;         /* set after camera init succeeds */
    bool thermal_override;     /* set when hard-coded thermal guardrail fires */
} edge_rl_state_t;

/* Declared in app_main.c */
extern edge_rl_state_t g_state;
extern SemaphoreHandle_t g_state_mutex;
