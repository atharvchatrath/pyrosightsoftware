#include "buttons.h"

#include "button_fsm.h"
#include "driver/gpio.h"
#include "esp_check.h"

static const char *TAG = "buttons";

static buttons_config_t C;
static btn_fsm_t fsm[2];

esp_err_t buttons_init(const buttons_config_t *cfg)
{
    C = *cfg;
    uint64_t mask = 0;
    if (cfg->pin_a >= 0) mask |= 1ULL << cfg->pin_a;
    if (cfg->pin_b >= 0) mask |= 1ULL << cfg->pin_b;
    gpio_config_t io = {
        .pin_bit_mask = mask,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = cfg->active_low ? GPIO_PULLUP_ENABLE : GPIO_PULLUP_DISABLE,
        .pull_down_en = cfg->active_low ? GPIO_PULLDOWN_DISABLE : GPIO_PULLDOWN_ENABLE,
    };
    if (mask) ESP_RETURN_ON_ERROR(gpio_config(&io), TAG, "gpio");
    btn_fsm_init(&fsm[0], cfg->debounce_ms, cfg->long_ms);
    btn_fsm_init(&fsm[1], cfg->debounce_ms, cfg->long_ms);
    return ESP_OK;
}

static bool pressed(int pin)
{
    if (pin < 0) return false;
    int l = gpio_get_level(pin);
    return C.active_low ? l == 0 : l != 0;
}

int buttons_poll(uint32_t t_ms, ps_button_t out[2])
{
    int n = 0;
    btn_event_t a = btn_fsm_update(&fsm[0], pressed(C.pin_a), t_ms);
    btn_event_t b = btn_fsm_update(&fsm[1], pressed(C.pin_b), t_ms);
    if (a == BTN_EV_SHORT) out[n++] = PS_BTN_WHERE_OUT;
    else if (a == BTN_EV_LONG) out[n++] = PS_BTN_MARK_ENTRY;
    if (b == BTN_EV_SHORT) out[n++] = PS_BTN_PALETTE;
    else if (b == BTN_EV_LONG) out[n++] = PS_BTN_BRIGHTNESS;
    return n;
}
