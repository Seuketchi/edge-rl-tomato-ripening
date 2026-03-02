/* ============================================================
 * edge_rl_policy.c — Pure-C MLP inference for distilled student
 *
 * Architecture: 16 → 64 (ReLU) → 64 (ReLU) → 3
 * Weights loaded from policy_weights.h (auto-generated)
 *
 * This replaces the previous stub with real inference.
 * No external ML library needed — just matmul + ReLU.
 * ============================================================ */

#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_err.h"
#include "app_config.h"
#include "policy_weights.h"

#define TAG_POLICY "POLICY"

/* ---- Forward pass buffers (static to avoid stack overflow) ---- */
static float buf_a[64];   /* max(LAYER0_OUT, LAYER1_OUT) */
static float buf_b[64];

/**
 * Dense layer: out[M] = W[M*N] @ in[N] + b[M], then optional ReLU.
 */
static void dense_layer(const float *w, const float *b,
                         const float *in, float *out,
                         int rows, int cols, int apply_relu)
{
    for (int i = 0; i < rows; i++) {
        float sum = b[i];
        const float *wi = &w[i * cols];
        for (int j = 0; j < cols; j++) {
            sum += wi[j] * in[j];
        }
        out[i] = apply_relu ? fmaxf(sum, 0.0f) : sum;
    }
}

/**
 * Argmax + softmax-max for action selection and confidence.
 */
static void argmax_with_conf(const float *logits, int n,
                              uint8_t *action, float *confidence)
{
    int best = 0;
    float best_val = logits[0];
    float sum_exp = 0.0f;

    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }

    /* Softmax for confidence (numerically stable) */
    for (int i = 0; i < n; i++) {
        sum_exp += expf(logits[i] - best_val);
    }

    *action = (uint8_t)best;
    *confidence = 1.0f / sum_exp;  /* = exp(best - best) / sum = 1/sum */
}

esp_err_t edge_rl_policy_infer(const float *state, int state_dim,
                                uint8_t *out_action, float *out_confidence)
{
    if (state == NULL || out_action == NULL || out_confidence == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (state_dim != STUDENT_STATE_DIM) {
        ESP_LOGE(TAG_POLICY, "State dim mismatch: expected %d, got %d",
                 STUDENT_STATE_DIM, state_dim);
        return ESP_ERR_INVALID_SIZE;
    }

    /* Layer 0: input[16] → hidden1[64] + ReLU */
    dense_layer(w0, b0, state, buf_a, LAYER0_OUT, LAYER0_IN, 1);

    /* Layer 1: hidden1[64] → hidden2[64] + ReLU */
    dense_layer(w1, b1, buf_a, buf_b, LAYER1_OUT, LAYER1_IN, 1);

    /* Layer 2: hidden2[64] → logits[3] (no activation) */
    float logits[STUDENT_ACTION_DIM];
    dense_layer(w2, b2, buf_b, logits, LAYER2_OUT, LAYER2_IN, 0);

    /* Argmax + confidence */
    argmax_with_conf(logits, STUDENT_ACTION_DIM, out_action, out_confidence);

    return ESP_OK;
}
