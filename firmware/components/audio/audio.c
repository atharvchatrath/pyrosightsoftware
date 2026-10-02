#include "audio.h"

#include <string.h>

#include "audio_pack.h"
#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "audio";

#define CHUNK_SAMPLES 512       /* 32 ms: abort checks happen at this granularity */
#define GAP_MS 70               /* between the parts of one message */
#define BEEP_AMPLITUDE 9000

static struct {
    audio_config_t cfg;
    i2s_chan_handle_t tx;
    esp_codec_dev_handle_t codec;
    const esp_partition_t *part;
    const uint8_t *map;         /* mmapped partition */
    esp_partition_mmap_handle_t map_handle;
    audio_pack_t pack;
    audio_stats_t st;
    int16_t buf[CHUNK_SAMPLES];
    bool ready;
} A;

static esp_err_t write_samples(const int16_t *s, size_t n)
{
    int r = esp_codec_dev_write(A.codec, (void *)s, (int)(n * sizeof(int16_t)));
    if (r != ESP_CODEC_DEV_OK) {
        A.st.write_errors++;
        return ESP_FAIL;
    }
    return ESP_OK;
}

static void write_silence(uint32_t ms)
{
    memset(A.buf, 0, sizeof(A.buf));
    uint32_t n = ms * AUDIO_SAMPLE_RATE / 1000;
    while (n) {
        uint32_t k = n > CHUNK_SAMPLES ? CHUNK_SAMPLES : n;
        if (write_samples(A.buf, k) != ESP_OK) return;
        n -= k;
    }
}

static void load_pack(void)
{
    A.part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY,
                                      A.cfg.partition_label ? A.cfg.partition_label : "audio");
    if (!A.part) {
        ESP_LOGW(TAG, "no \"audio\" partition: beeps only");
        A.st.pack_err = AUDIO_PACK_ERR_SHORT;
        return;
    }
    esp_err_t e = esp_partition_mmap(A.part, 0, A.part->size, ESP_PARTITION_MMAP_DATA,
                                     (const void **)&A.map, &A.map_handle);
    if (e != ESP_OK) {
        ESP_LOGW(TAG, "mmap audio partition failed (%s): beeps only", esp_err_to_name(e));
        A.map = NULL;
        return;
    }
    const uint16_t count = audio_pack_peek_count(A.map, AUDIO_PACK_HDR_BYTES);
    int r = count ? audio_pack_parse(A.map, audio_pack_index_bytes(count), A.part->size, &A.pack)
                  : AUDIO_PACK_ERR_MAGIC;
    A.st.pack_ok = (r == AUDIO_PACK_OK);
    A.st.pack_err = r;
    A.st.clips_available = A.pack.n_valid;
    if (r != AUDIO_PACK_OK)
        ESP_LOGW(TAG, "audio partition invalid (err %d, erased?): beeps only", r);
    else if (A.pack.count != PS_PHRASE_COUNT)
        ESP_LOGW(TAG, "audio pack has %u entries, firmware expects %d: rebuild with tools/make_audio_clips.py",
                 A.pack.count, PS_PHRASE_COUNT);
    else
        ESP_LOGI(TAG, "audio pack: %u of %d phrases", A.pack.n_valid, PS_PHRASE_COUNT - 1);
}

esp_err_t audio_init(const audio_config_t *cfg)
{
    A.cfg = *cfg;

    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG((i2s_port_t)cfg->i2s_port, I2S_ROLE_MASTER);
    cc.auto_clear = true;   /* output silence on underrun instead of repeating the last buffer */
    ESP_RETURN_ON_ERROR(i2s_new_channel(&cc, &A.tx, NULL), TAG, "i2s channel");
    i2s_std_config_t sc = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),   /* MCLK = 256 fs = 4.096 MHz */
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = cfg->pin_mclk,
            .bclk = cfg->pin_bclk,
            .ws = cfg->pin_ws,
            .dout = cfg->pin_dout,
            .din = I2S_GPIO_UNUSED,
        },
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(A.tx, &sc), TAG, "i2s std");
    ESP_RETURN_ON_ERROR(i2s_channel_enable(A.tx), TAG, "i2s enable");

    /* ES8311 through esp_codec_dev (uses the new I2C master driver bus handle,
     * so it can share the bus with other new-driver devices). */
    audio_codec_i2s_cfg_t i2s_cfg = { .port = (uint8_t)cfg->i2s_port, .rx_handle = NULL, .tx_handle = A.tx };
    const audio_codec_data_if_t *data_if = audio_codec_new_i2s_data(&i2s_cfg);
    audio_codec_i2c_cfg_t i2c_cfg = {
        .port = 0,                                   /* ignored when bus_handle is set */
        .addr = (uint8_t)(cfg->codec_addr_7bit << 1),/* esp_codec_dev takes the 8-bit address */
        .bus_handle = cfg->i2c_bus,
    };
    const audio_codec_ctrl_if_t *ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
    const audio_codec_gpio_if_t *gpio_if = audio_codec_new_gpio();
    ESP_RETURN_ON_FALSE(data_if && ctrl_if && gpio_if, ESP_FAIL, TAG, "codec interfaces");
    es8311_codec_cfg_t es = {
        .ctrl_if = ctrl_if,
        .gpio_if = gpio_if,
        .codec_mode = ESP_CODEC_DEV_WORK_MODE_DAC,
        .pa_pin = (int16_t)cfg->pin_pa,
        .pa_reverted = false,
        .master_mode = false,
        .use_mclk = true,
        .hw_gain = { .pa_voltage = 5.0f, .codec_dac_voltage = 3.3f },
    };
    const audio_codec_if_t *codec_if = es8311_codec_new(&es);
    ESP_RETURN_ON_FALSE(codec_if, ESP_FAIL, TAG, "es8311 not responding");
    esp_codec_dev_cfg_t dc = { .dev_type = ESP_CODEC_DEV_TYPE_OUT, .codec_if = codec_if, .data_if = data_if };
    A.codec = esp_codec_dev_new(&dc);
    ESP_RETURN_ON_FALSE(A.codec, ESP_FAIL, TAG, "codec dev");
    esp_codec_dev_sample_info_t fs = {
        .bits_per_sample = 16,
        .channel = 1,
        .sample_rate = AUDIO_SAMPLE_RATE,
    };
    ESP_RETURN_ON_FALSE(esp_codec_dev_open(A.codec, &fs) == ESP_CODEC_DEV_OK, ESP_FAIL, TAG, "codec open");
    esp_codec_dev_set_out_vol(A.codec, cfg->volume);

    load_pack();
    A.ready = true;
    return ESP_OK;
}

esp_err_t audio_set_volume(uint8_t v)
{
    if (!A.ready) return ESP_ERR_INVALID_STATE;
    return esp_codec_dev_set_out_vol(A.codec, v) == ESP_CODEC_DEV_OK ? ESP_OK : ESP_FAIL;
}

/* Returns false if aborted. */
static bool play_beeps(int prio, audio_abort_fn ab, void *ctx)
{
    audio_tone_t tones[AUDIO_BEEP_MAX_TONES];
    int n = audio_beep_pattern(prio, tones);
    audio_tone_gen_t g;
    audio_tone_gen_init(&g, tones, n, BEEP_AMPLITUDE);
    size_t k;
    A.st.beeps++;
    while ((k = audio_tone_gen_fill(&g, A.buf, CHUNK_SAMPLES)) > 0) {
        if (write_samples(A.buf, k) != ESP_OK) return true;
        if (ab && ab(ctx, prio)) return false;
    }
    return true;
}

static bool play_clip(uint32_t off, uint32_t len, int prio, audio_abort_fn ab, void *ctx)
{
    const int16_t *s = (const int16_t *)(A.map + off);   /* 4-byte aligned by the packer */
    size_t n = len / 2;
    A.st.clips++;
    while (n) {
        size_t k = n > CHUNK_SAMPLES ? CHUNK_SAMPLES : n;
        if (write_samples(s, k) != ESP_OK) return true;
        s += k;
        n -= k;
        if (ab && ab(ctx, prio)) return false;
    }
    return true;
}

esp_err_t audio_play_alert(const ps_alert_t *a, audio_abort_fn ab, void *ctx)
{
    if (!A.ready) return ESP_ERR_INVALID_STATE;
    A.st.alerts++;
    bool beeped = false;
    for (int i = 0; i < a->n_parts; i++) {
        uint32_t off, len;
        bool ok;
        if (A.map && A.st.pack_ok && audio_pack_clip(&A.pack, a->parts[i], &off, &len)) {
            ok = play_clip(off, len, a->prio, ab, ctx);
        } else {
            A.st.missing_clips++;
            /* One beep pattern per message is enough: the parts of a message
             * would otherwise turn into a long alarm. */
            ok = beeped ? true : play_beeps(a->prio, ab, ctx);
            beeped = true;
        }
        if (!ok) {
            A.st.aborted++;
            write_silence(GAP_MS);
            return ESP_ERR_INVALID_STATE;
        }
        write_silence(GAP_MS);
    }
    return ESP_OK;
}

esp_err_t audio_beep(int prio)
{
    if (!A.ready) return ESP_ERR_INVALID_STATE;
    play_beeps(prio, NULL, NULL);
    write_silence(GAP_MS);
    return ESP_OK;
}

void audio_get_stats(audio_stats_t *out) { *out = A.st; }
