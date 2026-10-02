/*
 * ESP-DL v3 wrapper. All ESP-DL specifics are confined to this file.
 */
#include "detector.h"

#include <cstring>
#include <map>
#include <string>

#include "detector_quant.h"
#include "dl_model_base.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"

static const char *TAG = "detector";

namespace {

struct Out {
    dl::TensorBase *t = nullptr;
    dq_layout_t layout = DQ_LAYOUT_UNKNOWN;
};

struct State {
    dl::Model *model = nullptr;
    dl::TensorBase *input = nullptr;
    Out heat, wh, off;
    int8_t lut[256];
    float *f_heat = nullptr, *f_wh = nullptr, *f_off = nullptr;  /* [2][30][40] each */
    detector_stats_t st = {};
};

State S;

constexpr int kPlane = DETECTOR_GRID_W * DETECTOR_GRID_H;
constexpr int kCh = 2;

bool name_has(const std::string &n, const char *a, const char *b)
{
    return n.find(a) != std::string::npos || (b && n.find(b) != std::string::npos);
}

bool partition_looks_programmed(const char *label)
{
    const esp_partition_t *p =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label);
    if (!p) {
        ESP_LOGW(TAG, "no \"%s\" partition", label);
        return false;
    }
    uint8_t head[16];
    if (esp_partition_read(p, 0, head, sizeof(head)) != ESP_OK) return false;
    bool erased = true, zero = true;
    for (uint8_t b : head) {
        if (b != 0xFF) erased = false;
        if (b != 0x00) zero = false;
    }
    if (erased || zero) {
        ESP_LOGW(TAG, "\"%s\" partition is empty: classical detector only", label);
        return false;
    }
    return true;
}

bool shape_of(dl::TensorBase *t, int *s, int *nd)
{
    *nd = (int)t->shape.size();
    if (*nd > 4 || *nd < 3) return false;
    for (int i = 0; i < *nd; i++) s[i] = t->shape[i];
    return true;
}

bool bind_output(Out &o, dl::TensorBase *t, const std::string &name)
{
    int s[4], nd;
    if (!shape_of(t, s, &nd)) return false;
    o.layout = dq_layout(s, nd, kCh, DETECTOR_GRID_H, DETECTOR_GRID_W);
    if (o.layout == DQ_LAYOUT_UNKNOWN) {
        ESP_LOGE(TAG, "output %s: unexpected shape", name.c_str());
        return false;
    }
    if (t->dtype != dl::DATA_TYPE_INT8 && t->dtype != dl::DATA_TYPE_INT16) {
        ESP_LOGE(TAG, "output %s: unsupported dtype", name.c_str());
        return false;
    }
    o.t = t;
    return true;
}

void dequant(const Out &o, float *dst)
{
    if (o.t->dtype == dl::DATA_TYPE_INT8)
        dq_dequant_i8_to_chw(static_cast<const int8_t *>(o.t->data), o.layout, kCh, DETECTOR_GRID_H,
                             DETECTOR_GRID_W, o.t->exponent, dst);
    else
        dq_dequant_i16_to_chw(static_cast<const int16_t *>(o.t->data), o.layout, kCh, DETECTOR_GRID_H,
                              DETECTOR_GRID_W, o.t->exponent, dst);
}

void unload()
{
    delete S.model;
    S.model = nullptr;
    S.input = nullptr;
    S.st.loaded = false;
}

}  // namespace

extern "C" esp_err_t detector_init(const char *label)
{
    if (!partition_looks_programmed(label)) return ESP_ERR_NOT_FOUND;

    S.f_heat = static_cast<float *>(heap_caps_malloc(3 * kCh * kPlane * sizeof(float), MALLOC_CAP_8BIT));
    if (!S.f_heat) return ESP_ERR_NO_MEM;
    S.f_wh = S.f_heat + kCh * kPlane;
    S.f_off = S.f_wh + kCh * kPlane;

    /* NOTE: ESP-DL validates the FlatBuffers model while loading; a corrupt
     * (not merely erased) partition may still abort inside ESP-DL. Erase the
     * partition (tools/flash_partitions.sh --erase-model) to disable the NN. */
    S.model = new dl::Model(label, fbs::MODEL_LOCATION_IN_FLASH_PARTITION);
    auto inputs = S.model->get_inputs();
    auto outputs = S.model->get_outputs();
    if (inputs.size() != 1 || outputs.size() != 3) {
        ESP_LOGE(TAG, "model has %u inputs / %u outputs, expected 1 / 3", (unsigned)inputs.size(),
                 (unsigned)outputs.size());
        unload();
        return ESP_ERR_INVALID_RESPONSE;
    }
    S.input = inputs.begin()->second;
    int s[4], nd;
    if (!shape_of(S.input, s, &nd) || S.input->dtype != dl::DATA_TYPE_INT8 ||
        S.input->get_size() != PS_THERM_PIXELS) {
        ESP_LOGE(TAG, "input must be int8 1x120x160x1");
        unload();
        return ESP_ERR_INVALID_RESPONSE;
    }
    bool ok = true;
    for (auto &kv : outputs) {
        const std::string &n = kv.first;
        if (name_has(n, "heat", "hm")) ok &= bind_output(S.heat, kv.second, n);
        else if (name_has(n, "wh", "size")) ok &= bind_output(S.wh, kv.second, n);
        else if (name_has(n, "off", "reg")) ok &= bind_output(S.off, kv.second, n);
        else ESP_LOGW(TAG, "unrecognised output name \"%s\"", n.c_str());
    }
    if (!ok || !S.heat.t || !S.wh.t || !S.off.t) {
        ESP_LOGE(TAG, "outputs must be named heat/wh/off with shape 2x30x40");
        unload();
        return ESP_ERR_INVALID_RESPONSE;
    }

    S.st.input_exponent = S.input->exponent;
    dq_build_input_lut(S.input->exponent, S.lut);
    S.st.loaded = true;
    ESP_LOGI(TAG, "model loaded: input exponent %d, heat/wh/off exponents %d/%d/%d", S.input->exponent,
             S.heat.t->exponent, S.wh.t->exponent, S.off.t->exponent);
    return ESP_OK;
}

extern "C" bool detector_ready(void) { return S.st.loaded; }

extern "C" esp_err_t detector_run(const ps_gray_frame_t *in, float thr, uint32_t deadline_us,
                                  ps_detections_t *out)
{
    if (!S.st.loaded) return ESP_ERR_INVALID_STATE;
    const int64_t t0 = esp_timer_get_time();
    dq_quantize_input(in->px, PS_THERM_PIXELS, S.lut, static_cast<int8_t *>(S.input->data));
    S.model->run();
    dequant(S.heat, S.f_heat);
    dequant(S.wh, S.f_wh);
    dequant(S.off, S.f_off);
    std::memset(out, 0, sizeof(*out));
    ps_centernet_decode(S.f_heat, S.f_wh, S.f_off, DETECTOR_GRID_W, DETECTOR_GRID_H, DETECTOR_STRIDE, thr, out);
    const uint32_t us = (uint32_t)(esp_timer_get_time() - t0);
    out->source = PS_DETECTOR_NEURAL;
    out->latency_us = us;

    S.st.runs++;
    S.st.last_us = us;
    if (us > S.st.max_us) S.st.max_us = us;
    S.st.avg_us = S.st.runs == 1 ? (float)us : S.st.avg_us * 0.9f + 0.1f * (float)us;
    if (deadline_us && us > deadline_us) {
        S.st.overruns++;
        return ESP_ERR_TIMEOUT;
    }
    return ESP_OK;
}

extern "C" void detector_get_stats(detector_stats_t *o) { *o = S.st; }
