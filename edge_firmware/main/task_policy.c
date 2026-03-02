/* ============================================================
 * task_policy.c — RL policy inference task
 *
 * Assembles a 16-dimensional state vector (Variant B) from g_state
 * and runs the distilled student MLP (INT8) via components/edge_rl_policy.
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
 *
 * Safety: A hard-coded thermal guardrail overrides the policy
 * output if temperature exits the [12.5, 35] °C safe band.
 *
 * NOTE: edge_rl_policy_infer is stubbed until policy_data.h is
 * generated from the export pipeline (Step 4).
 * ============================================================ */

#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "app_config.h"
#include "shared_state.h"

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

/* ---- Online k₁ adaptation via EMA ---- */
#define K1_EMA_ALPHA  0.05f     /* Smoothing factor (lower = more stable) */
#define K1_MIN        0.01f     /* Floor to prevent k₁ collapsing */
#define K1_MAX        0.20f     /* Ceiling to prevent runaway estimates */
static float s_k1_estimate = K1_DEFAULT;

/* ---- Temporal buffer for dX/dt ---- */
static float s_x_buf[CHROMATIC_VEL_BUF_SIZE] = {0};
static int   s_x_buf_idx  = 0;
static int   s_x_buf_fill = 0;    /* how many valid entries */

/**
 * Push a new Chromatic Index reading into the circular buffer
 * and return the finite-difference velocity dX/dt.
 */
static float compute_dx_dt(float x_new)
{
    float dx_dt = 0.0f;

    if (s_x_buf_fill > 0) {
        /* Oldest valid sample */
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
 * Update k₁ estimate from observed dX/dt using Exponential Moving Average.
 * From the ODE dX/dt = -k₁(T - T_base)X, we solve:
 *   k₁_obs = -dX/dt / ((T - T_base) * X)
 * Only updates when the denominator is large enough to avoid noise.
 */
static void update_k1_estimate(float dx_dt, float x, float temperature)
{
    float t_diff = temperature - SAFETY_TEMP_MIN;  /* T - T_base */
    float denom  = t_diff * x;

    /* Only update if denominator is meaningful (avoid div-by-zero/noise) */
    if (denom > 0.5f && x > 0.1f) {
        float k1_obs = -dx_dt / denom;
        if (k1_obs > K1_MIN && k1_obs < K1_MAX) {
            s_k1_estimate = (1.0f - K1_EMA_ALPHA) * s_k1_estimate
                          + K1_EMA_ALPHA * k1_obs;
        }
    }
}

/**
 * Analytical ODE solution for the reference trajectory (ROYG decay):
 *   X_ref(t) = exp(-k1 * (T_ideal - T_base) * t_days)
 * X_ref decays from 1.0 (Green) toward 0.0 (Red).
 * Uses adaptive k₁ estimate when available.
 */
static float compute_x_ref(float hours_elapsed)
{
    float t_days = hours_elapsed / 24.0f;
    float exponent = -s_k1_estimate * (T_IDEAL - SAFETY_TEMP_MIN) * t_days;
    return expf(exponent);
}

void edge_rl_task_policy(void *pvParameters)
{
    ESP_LOGI(TAG, "[policy] Task started on core %d", xPortGetCoreID());

    float state_vec[POLICY_STATE_DIM] = {0};
    uint32_t day = 0;

    /* Run policy every time vision produces a new result */
    while (1) {
        /* Snapshot current state under mutex */
        bool  vision_valid = false;
        float temperature  = 0.0f;

        if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            vision_valid = g_state.vision_valid;
            if (vision_valid) {
                /* --- Read raw fields --- */
                float x         = g_state.chromatic_x;
                float dx_dt     = compute_dx_dt(x);
                /* Online k₁ adaptation from observed ripening rate */
                update_k1_estimate(dx_dt, x, g_state.temperature_c);
                float days_elap = (float)g_state.day_counter;
                float target_day = 5.0f;  /* mid-range of [3,7] target window */
                float t_rem     = target_day - days_elap;  /* remaining days */
                float hours_elap = days_elap * 24.0f;
                float x_ref     = compute_x_ref(hours_elap);

                temperature = g_state.temperature_c;

                /* ---- Assemble 16D state vector (Variant B) ----
                 * [0]  X            [1] dX/dt       [2] X_ref
                 * [3]  C_mu_R       [4] C_mu_G      [5] C_mu_B
                 * [6]  C_sig_R      [7] C_sig_G     [8] C_sig_B
                 * [9]  C_mode_R     [10] C_mode_G   [11] C_mode_B
                 * [12] T            [13] H           [14] t_e
                 * [15] t_rem                                     */
                state_vec[0]  = x;
                state_vec[1]  = dx_dt;
                state_vec[2]  = x_ref;
                /* C_mu: RGB channel means from vision task */
                state_vec[3]  = g_state.rgb_mean[0];  /* C_mu_R */
                state_vec[4]  = g_state.rgb_mean[1];  /* C_mu_G */
                state_vec[5]  = g_state.rgb_mean[2];  /* C_mu_B */
                /* C_sig: RGB channel std devs from vision task */
                state_vec[6]  = g_state.rgb_std[0];   /* C_sig_R */
                state_vec[7]  = g_state.rgb_std[1];   /* C_sig_G */
                state_vec[8]  = g_state.rgb_std[2];   /* C_sig_B */
                /* C_mode: RGB channel modes from vision task */
                state_vec[9]  = g_state.rgb_mode[0];  /* C_mode_R */
                state_vec[10] = g_state.rgb_mode[1];  /* C_mode_G */
                state_vec[11] = g_state.rgb_mode[2];  /* C_mode_B */
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

        int64_t latency_ms = (esp_timer_get_time() - t_start) / 1000;

        if (err != ESP_OK) {
            ESP_LOGW(TAG, "[policy] Inference failed: %s", esp_err_to_name(err));
            vTaskDelay(pdMS_TO_TICKS(CAPTURE_INTERVAL_MS));
            continue;
        }

        /* ============================================================
         * CONFIDENCE-BASED ACTION GATING
         * If the policy is not confident enough, default to the safest
         * action (MAINTAIN) to avoid unnecessary actuation.
         * ============================================================ */
#define CONFIDENCE_THRESHOLD 0.6f
        if (confidence < CONFIDENCE_THRESHOLD) {
            ESP_LOGI(TAG, "[policy] Low confidence %.3f < %.1f — defaulting to MAINTAIN",
                     confidence, CONFIDENCE_THRESHOLD);
            action = ACTION_MAINTAIN;
        }

        /* ============================================================
         * HARD-CODED THERMAL GUARDRAIL
         * Overrides the RL policy output to protect biological payload.
         * This runs INDEPENDENT of the learned policy.
         * ============================================================ */
        bool override = false;
        if (temperature > SAFETY_TEMP_MAX) {
            /* Temperature ceiling breached — force heater OFF / cool */
            if (action == ACTION_HEAT) {
                action = ACTION_COOL;
                override = true;
                ESP_LOGW(TAG, "[GUARDRAIL] T=%.1f > %.1f°C — overriding HEAT→COOL",
                         temperature, SAFETY_TEMP_MAX);
            }
        } else if (temperature < SAFETY_TEMP_MIN) {
            /* Temperature floor breached — force fan OFF / heat */
            if (action == ACTION_COOL) {
                action = ACTION_HEAT;
                override = true;
                ESP_LOGW(TAG, "[GUARDRAIL] T=%.1f < %.1f°C — overriding COOL→HEAT",
                         temperature, SAFETY_TEMP_MIN);
            }
        }

        ESP_LOGI(TAG, "[policy] Day=%lu X=%.3f action=%s conf=%.3f lat=%lldms%s",
                 (unsigned long)day, state_vec[0],
                 ACTION_NAMES[action], confidence, latency_ms,
                 override ? " [OVERRIDE]" : "");

        /* Write action back to shared state */
        if (xSemaphoreTake(g_state_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
            g_state.action            = action;
            g_state.action_confidence = confidence;
            g_state.thermal_override  = override;
            /* Harvest is auto post-processing: triggered when X <= ripe threshold
             * in the vision task, NOT by policy action. */
            xSemaphoreGive(g_state_mutex);
        }

        vTaskDelay(pdMS_TO_TICKS(CAPTURE_INTERVAL_MS));
    }
}
