#pragma once

/* ============================================================
 * app_config.h — Edge-RL: Global constants and configuration
 * All magic numbers go here. Never hardcode values in .c files.
 * ============================================================ */

/* ---- Task priorities (higher = more urgent) ---- */
#define TASK_PRIORITY_CAMERA        3
#define TASK_PRIORITY_SENSORS       4
#define TASK_PRIORITY_VISION        2
#define TASK_PRIORITY_POLICY        2
#define TASK_PRIORITY_TELEMETRY     1

/* ---- Task stack sizes (words, not bytes) ---- */
#define TASK_STACK_CAMERA           4096
#define TASK_STACK_SENSORS          2048
#define TASK_STACK_VISION           8192   /* larger: runs model inference */
#define TASK_STACK_POLICY           8192   /* larger: MLP inference + golden test + ODE sim */
#define TASK_STACK_TELEMETRY        4096   /* JSON formatting needs space */

/* ---- Timing ---- */
#define CAPTURE_INTERVAL_MS         (30 * 60 * 1000)  /* 30 minutes */
#define SENSOR_INTERVAL_MS          (60 * 1000)        /* 1 minute   */
#define TELEMETRY_INTERVAL_MS       (5  * 1000)        /* 5 seconds  */

/* ---- Camera (OV2640) ---- */
#define CAM_PIN_PWDN                -1
#define CAM_PIN_RESET               -1
#define CAM_PIN_XCLK                21
#define CAM_PIN_SIOD                26
#define CAM_PIN_SIOC                27
#define CAM_PIN_D7                  35
#define CAM_PIN_D6                  34
#define CAM_PIN_D5                  39
#define CAM_PIN_D4                  36
#define CAM_PIN_D3                  19
#define CAM_PIN_D2                  18
#define CAM_PIN_D1                   5
#define CAM_PIN_D0                   4
#define CAM_PIN_VSYNC               25
#define CAM_PIN_HREF                23
#define CAM_PIN_PCLK                22
#define CAM_XCLK_FREQ_HZ            20000000
#define CAM_FRAME_SIZE              FRAMESIZE_96X96  /* matches model input */

/* ---- DHT22 sensor ---- */
#define DHT22_GPIO                  GPIO_NUM_32

/* ---- Vision model ---- */
#define VISION_INPUT_W              96
#define VISION_INPUT_H              96
#define VISION_NUM_CLASSES          4
#define VISION_CONFIDENCE_THRESH    0.70f  /* minimum confidence to act */
#define CLASS_UNRIPE                0
#define CLASS_RIPE                  1
#define CLASS_OLD                   2
#define CLASS_DAMAGED               3

/* ---- RL policy ---- */
#define POLICY_NUM_ACTIONS          3   /* maintain, heat(+ΔT), cool(heater off) */
#define POLICY_STATE_DIM            16  /* Variant B: X, dX/dt, X_ref, C_mu(3), C_sig(3), C_mode(3), T, H, t_e, t_rem */

/* ---- Power Profiling ---- */
#define PROFILING_PIN_ML            GPIO_NUM_12  /* toggles HIGH during active ML inference */

/* ---- Safety guardrail constants ---- */
#define SAFETY_TEMP_MAX             35.0f  /* °C — hard-coded thermal ceiling */
#define SAFETY_TEMP_MIN             12.5f  /* °C — hard-coded thermal floor  */
#define CHROMATIC_VEL_BUF_SIZE      3      /* rolling buffer for dX/dt calc */

/* ---- Policy action labels ---- */
#define ACTION_MAINTAIN             0
#define ACTION_HEAT                 1      /* Heater relay ON */
#define ACTION_COOL                 2      /* Heater relay OFF (passive cooling toward ambient) */

/* ---- Logging tag ---- */
#define TAG                         "EDGE_RL"
