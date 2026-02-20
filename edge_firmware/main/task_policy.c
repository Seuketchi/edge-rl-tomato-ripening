/* ============================================================
 * task_policy.c — RL policy inference task
 *
 * Assembles a 16-dimensional state vector (Variant B) from g_state
 * and runs the distilled student MLP via components/edge_rl_policy.
 *
 * At boot:
 *   1. Runs golden vector test (20 known states vs expected actions)
 *   2. Runs on-device ODE simulation (36-step episode)
 *   3. Enters main policy loop
 *
 * State vector S ∈ R^16:
 *   [X, dX/dt, X_ref, C_mu_R, C_mu_G, C_mu_B,
 *    C_sig_R, C_sig_G, C_sig_B, C_mode_R, C_mode_G, C_mode_B,
 *    T, H, t_e, t_rem]
 *
 * Heater-only actuation (no active cooling hardware):
 *   0 = MAINTAIN — heater stays in current state
 *   1 = HEAT     — heater relay ON (+ΔT setpoint increment)
 *   2 = COOL     — heater relay OFF (passive cooling toward ambient)
 * ============================================================ */

#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "app_config.h"
#include "shared_state.h"
#include "golden_vectors.h"

/* From components/edge_rl_policy */
esp_err_t edge_rl_policy_infer(const float *state, int state_dim,
                                uint8_t *out_action, float *out_confidence);

/* Action label strings for logging */
static const char *ACTION_NAMES[POLICY_NUM_ACTIONS] = {
    "MAINTAIN", "HEAT", "COOL"
};

/* ---- ODE constants for X_ref computation ---- */
#define K1_DEFAULT    0.08f
#define T_IDEAL       20.0f     /* Reference temperature (°C) */

/* ---- Temporal buffer for dX/dt ---- */
static float s_x_buf[CHROMATIC_VEL_BUF_SIZE] = {0};
static int   s_x_buf_idx  = 0;
static int   s_x_buf_fill = 0;

/**
 * Push a new Chromatic Index reading into the circular buffer
 * and return the finite-difference velocity dX/dt.
 */
static float compute_dx_dt(float x_new)
{
    float dx_dt = 0.0f;

    if (s_x_buf_fill > 0) {
        int oldest = (s_x_buf_fill < CHROMATIC_VEL_BUF_SIZE)
                     ? 0
                     : (s_x_buf_idx + 1) % CHROMATIC_VEL_BUF_SIZE;
        float span = (float)(s_x_buf_fill < CHROMATIC_VEL_BUF_SIZE
                              ? s_x_buf_fill : CHROMATIC_VEL_BUF_SIZE);
        dx_dt = (x_new - s_x_buf[oldest]) / span;
    }

    s_x_buf[s_x_buf_idx] = x_new;
    s_x_buf_idx = (s_x_buf_idx + 1) % CHROMATIC_VEL_BUF_SIZE;
    if (s_x_buf_fill < CHROMATIC_VEL_BUF_SIZE) s_x_buf_fill++;

    return dx_dt;
}

/**
 * Analytical ODE solution for the reference trajectory (ROYG decay):
 *   X_ref(t) = exp(-k1 * (T_ideal - T_base) * t_days)
 */
static float compute_x_ref(float hours_elapsed)
{
    float t_days = hours_elapsed / 24.0f;
    float exponent = -K1_DEFAULT * (T_IDEAL - SAFETY_TEMP_MIN) * t_days;
    return expf(exponent);
}


/* ============================================================
 *  GOLDEN VECTOR TEST
 *  Run 20 known (state → action) pairs through inference
 *  and verify the outputs match Python.
 * ============================================================ */
static int run_golden_test(void)
{
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "╔══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║     GOLDEN VECTOR TEST (20 vectors)  ║");
    ESP_LOGI(TAG, "╚══════════════════════════════════════╝");

    int pass = 0;
    int fail = 0;

    for (int i = 0; i < NUM_GOLDEN_VECTORS; i++) {
        uint8_t action = 0;
        float conf = 0.0f;
        esp_err_t err = edge_rl_policy_infer(
            golden_states[i], GOLDEN_STATE_DIM, &action, &conf);

        if (err != ESP_OK) {
            ESP_LOGE(TAG, "  [GOLDEN %2d] Inference error: %s",
                     i, esp_err_to_name(err));
            fail++;
            continue;
        }

        if (action == golden_actions[i]) {
            pass++;
            ESP_LOGI(TAG, "  [GOLDEN %2d] ✓ action=%s conf=%.3f",
                     i, ACTION_NAMES[action], conf);
        } else {
            fail++;
            ESP_LOGW(TAG, "  [GOLDEN %2d] ✗ expected=%s got=%s conf=%.3f",
                     i, ACTION_NAMES[golden_actions[i]],
                     ACTION_NAMES[action], conf);
        }
    }

    ESP_LOGI(TAG, "");
    if (fail == 0) {
        ESP_LOGI(TAG, "  ════════════════════════════════════");
        ESP_LOGI(TAG, "  ✓ GOLDEN TEST PASSED: %d/%d", pass, NUM_GOLDEN_VECTORS);
        ESP_LOGI(TAG, "  ════════════════════════════════════");
    } else {
        ESP_LOGW(TAG, "  ════════════════════════════════════");
        ESP_LOGW(TAG, "  ✗ GOLDEN TEST FAILED: %d/%d passed", pass, NUM_GOLDEN_VECTORS);
        ESP_LOGW(TAG, "  ════════════════════════════════════");
    }
    ESP_LOGI(TAG, "");

    return fail;
}


/* ============================================================
 *  ON-DEVICE ODE SIMULATION
 *  Runs a complete ripening episode using the embedded policy
 *  and prints the trajectory over serial.
 * ============================================================ */

/* Simulation constants matching ml_training/rl/simulator.py */
#define SIM_K1       0.08f
#define SIM_K2       0.025f
#define SIM_T_BASE   12.5f
#define SIM_T_AMB    25.0f
#define SIM_DT_HOURS 1.0f       /* 1 hour per step */
#define SIM_STEPS    36          /* 1.5 days */
#define SIM_TARGET_DAY 5.0f
#define SIM_RIPE_THRESH 0.3f    /* X <= this → ripe */
#define SIM_DT_EFFECT    1.5f   /* temperature delta per action step */

static void run_ondevice_sim(void)
{
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "╔══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   ON-DEVICE ODE SIMULATION (36 steps)║");
    ESP_LOGI(TAG, "╚══════════════════════════════════════╝");

    /* Initial conditions */
    float X = 0.90f;          /* Green tomato */
    float T = SIM_T_AMB;      /* Ambient temperature */
    float H = 65.0f;          /* Humidity % */
    float total_reward = 0.0f;
    int   harvest_step = -1;

    ESP_LOGI(TAG, "  %-4s  %-6s  %-6s  %-10s  %-6s", 
             "Step", "X", "T(°C)", "Action", "Conf");
    ESP_LOGI(TAG, "  ──── ────── ────── ────────── ──────");

    for (int step = 0; step < SIM_STEPS; step++) {
        float hours_elapsed = (float)step * SIM_DT_HOURS;
        float days_elapsed  = hours_elapsed / 24.0f;
        float t_rem         = SIM_TARGET_DAY - days_elapsed;
        float dx_dt         = 0.0f;  /* simplified for sim */

        /* Reference trajectory */
        float t_days = hours_elapsed / 24.0f;
        float x_ref  = expf(-SIM_K1 * (SIM_T_AMB - SIM_T_BASE) * t_days);

        /* Assemble 16D state vector (Variant B) */
        float state[POLICY_STATE_DIM];
        memset(state, 0, sizeof(state));
        state[0]  = X;              /* Chromatic index */
        state[1]  = dx_dt;          /* dX/dt */
        state[2]  = x_ref;          /* X_ref */
        /* state[3..11] = colour stats (zeros in sim) */
        state[12] = T;              /* Temperature */
        state[13] = H;              /* Humidity */
        state[14] = days_elapsed;   /* t_elapsed */
        state[15] = t_rem;          /* t_remaining */

        /* Run policy inference */
        uint8_t action = 0;
        float   conf   = 0.0f;
        esp_err_t err = edge_rl_policy_infer(state, POLICY_STATE_DIM,
                                              &action, &conf);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "  [SIM] Step %d: inference error", step);
            break;
        }

        ESP_LOGI(TAG, "  %-4d  %.3f   %.1f   %-10s  %.3f%s",
                 step, X, T, ACTION_NAMES[action], conf,
                 X <= SIM_RIPE_THRESH ? "  ← RIPE" : "");

        /* Apply action — temperature effect */
        float T_before = T;
        if (action == ACTION_HEAT) {
            T = fminf(T + SIM_DT_EFFECT, SAFETY_TEMP_MAX);
        } else if (action == ACTION_COOL) {
            T = fmaxf(T - SIM_DT_EFFECT, SIM_T_AMB - 2.0f);
        }
        /* else MAINTAIN: T stays */

        /* ODE step: dX/dt = -k1 * (T - T_base) * X + noise */
        float dX = -SIM_K1 * (T - SIM_T_BASE) * X * (SIM_DT_HOURS / 24.0f);
        X += dX;
        X = fmaxf(X, 0.0f);
        X = fminf(X, 1.0f);

        /* Simple reward: negative absolute tracking error */
        float reward = -fabsf(X - x_ref);
        total_reward += reward;

        /* Check harvest */
        if (harvest_step < 0 && X <= SIM_RIPE_THRESH) {
            harvest_step = step;
        }
    }

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "  ════════════════════════════════════");
    ESP_LOGI(TAG, "  ✓ Episode complete");
    ESP_LOGI(TAG, "    Final X:      %.3f %s", X,
             X <= SIM_RIPE_THRESH ? "(RIPE)" : "(still ripening)");
    ESP_LOGI(TAG, "    Final T:      %.1f °C", T);
    ESP_LOGI(TAG, "    Total reward: %+.2f", total_reward);
    if (harvest_step >= 0) {
        ESP_LOGI(TAG, "    Harvest step: %d (%.1f hours)",
                 harvest_step, harvest_step * SIM_DT_HOURS);
    }
    ESP_LOGI(TAG, "  ════════════════════════════════════");
    ESP_LOGI(TAG, "");
}


/* ============================================================
 *  MAIN POLICY TASK
 * ============================================================ */
void edge_rl_task_policy(void *pvParameters)
{
    ESP_LOGI(TAG, "[policy] Task started on core %d", xPortGetCoreID());

    /* ---- Run boot-time validation ---- */
    int golden_failures = run_golden_test();
    (void)golden_failures;  /* continue even if some fail */

    /* ---- Run on-device simulation ---- */
    run_ondevice_sim();

    /* ---- Enter main policy loop ---- */
    ESP_LOGI(TAG, "[policy] Entering main inference loop...");

    float state_vec[POLICY_STATE_DIM] = {0};
    uint32_t day = 0;

    while (1) {
        bool  vision_valid = false;
        float temperature  = 0.0f;

        if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            vision_valid = g_state.vision_valid;
            if (vision_valid) {
                float x         = g_state.chromatic_x;
                float dx_dt     = compute_dx_dt(x);
                float days_elap = (float)g_state.day_counter;
                float target_day = 5.0f;
                float t_rem     = target_day - days_elap;
                float hours_elap = days_elap * 24.0f;
                float x_ref     = compute_x_ref(hours_elap);

                temperature = g_state.temperature_c;

                state_vec[0]  = x;
                state_vec[1]  = dx_dt;
                state_vec[2]  = x_ref;
                state_vec[3]  = 0.0f;  state_vec[4]  = 0.0f;  state_vec[5]  = 0.0f;
                state_vec[6]  = 0.0f;  state_vec[7]  = 0.0f;  state_vec[8]  = 0.0f;
                state_vec[9]  = 0.0f;  state_vec[10] = 0.0f;  state_vec[11] = 0.0f;
                state_vec[12] = temperature;
                state_vec[13] = g_state.humidity_pct;
                state_vec[14] = days_elap;
                state_vec[15] = t_rem;

                day = g_state.day_counter;
            }
            xSemaphoreGive(g_state_mutex);
        }

        if (!vision_valid) {
            vTaskDelay(pdMS_TO_TICKS(5000));
            continue;
        }

        /* ---- Policy inference ---- */
        int64_t t_start = esp_timer_get_time();

        uint8_t action = ACTION_MAINTAIN;
        float   confidence = 0.0f;
        esp_err_t err = edge_rl_policy_infer(state_vec, POLICY_STATE_DIM,
                                              &action, &confidence);

        int64_t latency_us = esp_timer_get_time() - t_start;

        if (err != ESP_OK) {
            ESP_LOGW(TAG, "[policy] Inference failed: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(CAPTURE_INTERVAL_MS));
            continue;
        }

        /* ---- Thermal guardrail ---- */
        bool override = false;
        if (temperature > SAFETY_TEMP_MAX) {
            if (action == ACTION_HEAT) {
                action = ACTION_COOL;
                override = true;
                ESP_LOGW(TAG, "[GUARDRAIL] T=%.1f > %.1f°C — HEAT→COOL",
                         temperature, SAFETY_TEMP_MAX);
            }
        } else if (temperature < SAFETY_TEMP_MIN) {
            if (action == ACTION_COOL) {
                action = ACTION_HEAT;
                override = true;
                ESP_LOGW(TAG, "[GUARDRAIL] T=%.1f < %.1f°C — COOL→HEAT",
                         temperature, SAFETY_TEMP_MIN);
            }
        }

        ESP_LOGI(TAG, "[policy] Day=%lu X=%.3f action=%s conf=%.3f lat=%lldus%s",
                 (unsigned long)day, state_vec[0],
                 ACTION_NAMES[action], confidence, (long long)latency_us,
                 override ? " [OVERRIDE]" : "");

        /* Write action back to shared state */
        if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            g_state.action            = action;
            g_state.action_confidence = confidence;
            g_state.thermal_override  = override;
            xSemaphoreGive(g_state_mutex);
        }

        vTaskDelay(pdMS_TO_TICKS(CAPTURE_INTERVAL_MS));
    }
}
