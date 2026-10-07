/*
 * Parity test driver, native build: reads the command file written by
 * parity_test.js and runs it through nav_api.c (+ the device's ps_nav.c,
 * ps_config.c, ps_alerts.c) compiled with gcc. Prints one line per "P"
 * command: P <26 state floats> | <crumb floats> | <alert parts...>
 *
 *   Y yaw t | A mag t | S t | K t | M t | W t | R | P
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

void api_init(void);
void api_mark_entry(uint32_t t_ms);
void api_reset(void);
void api_on_yaw(float yaw_rad, uint32_t t_ms);
void api_on_step(uint32_t t_ms);
void api_on_accel(float mag, uint32_t t_ms);
void api_tick(uint32_t t_ms);
void api_where_out(uint32_t t_ms);
float *api_state(void);
float *api_crumbs(void);
int api_n_crumbs(void);
int api_alert_pop(void);
int32_t *api_alert_buf(void);

int main(int argc, char **argv)
{
    FILE *f = argc > 1 ? fopen(argv[1], "r") : stdin;
    if (!f) { perror("open"); return 1; }
    api_init();
    char line[256];
    while (fgets(line, sizeof line, f)) {
        char c = line[0];
        double v = 0; unsigned long t = 0;
        switch (c) {
        case 'Y': sscanf(line + 1, "%lf %lu", &v, &t); api_on_yaw((float)v, (uint32_t)t); break;
        case 'A': sscanf(line + 1, "%lf %lu", &v, &t); api_on_accel((float)v, (uint32_t)t); break;
        case 'S': sscanf(line + 1, "%lu", &t); api_on_step((uint32_t)t); break;
        case 'K': sscanf(line + 1, "%lu", &t); api_tick((uint32_t)t); break;
        case 'M': sscanf(line + 1, "%lu", &t); api_mark_entry((uint32_t)t); break;
        case 'W': sscanf(line + 1, "%lu", &t); api_where_out((uint32_t)t); break;
        case 'R': api_reset(); break;
        case 'P': {
            const float *s = api_state();
            printf("P");
            for (int i = 0; i < 26; i++) printf(" %.9g", s[i]);
            printf(" |");
            int n = api_n_crumbs();
            const float *cr = api_crumbs();
            for (int i = 0; i < 3 * n; i++) printf(" %.9g", cr[i]);
            printf(" |");
            int k;
            while ((k = api_alert_pop()) > 0) {
                const int32_t *b = api_alert_buf();
                for (int i = 0; i < k; i++) printf(" %d", b[i]);
                printf(" ;");
            }
            printf("\n");
            break;
        }
        default: break;
        }
    }
    return 0;
}
