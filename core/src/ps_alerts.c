#include "pyrosight/ps_alerts.h"

#include <math.h>
#include <string.h>

const char *const ps_phrase_text[PS_PHRASE_COUNT] = {
    [PS_PHRASE_NONE] = "",
    [PS_PHRASE_WAY_OUT_IS] = "Way out is",
    [PS_PHRASE_DIR_AHEAD] = "ahead.",
    [PS_PHRASE_DIR_AHEAD_LEFT] = "ahead, to your left.",
    [PS_PHRASE_DIR_LEFT] = "to your left.",
    [PS_PHRASE_DIR_BEHIND_LEFT] = "behind you, to the left.",
    [PS_PHRASE_DIR_BEHIND] = "behind you.",
    [PS_PHRASE_DIR_BEHIND_RIGHT] = "behind you, to the right.",
    [PS_PHRASE_DIR_RIGHT] = "to your right.",
    [PS_PHRASE_DIR_AHEAD_RIGHT] = "ahead, to your right.",
    [PS_PHRASE_NAV_UNRELIABLE] = "Navigation estimate unreliable.",
    [PS_PHRASE_FOLLOW_HOSE] = "Follow the hose line out.",
    [PS_PHRASE_NAV_RECOVERED] = "Navigation restored.",
    [PS_PHRASE_PERSON] = "Person",
    [PS_PHRASE_FIRE] = "Fire",
    [PS_PHRASE_CAMERA_LOST] = "Thermal camera lost.",
    [PS_PHRASE_CAMERA_RESTORED] = "Thermal camera restored.",
    [PS_PHRASE_MOTION_LOST] = "Motion sensor lost.",
    [PS_PHRASE_ENTRY_MARKED] = "Entry point marked.",
    [PS_PHRASE_AT_EXIT] = "You are at the entry point.",
    [PS_PHRASE_BATTERY_LOW] = "Battery low.",
    [PS_PHRASE_BATTERY_CRITICAL] = "Battery critical. Exit now.",
};

void ps_alerts_init(ps_alerts_t *a)
{
    memset(a, 0, sizeof(*a));
    a->level = PS_NAVCONF_GOOD;
}

void ps_alerts_push(ps_alerts_t *a, ps_alert_prio_t prio, uint32_t t_ms,
                    ps_phrase_t p0, ps_phrase_t p1, ps_phrase_t p2)
{
    ps_alert_t al;
    memset(&al, 0, sizeof(al));
    al.prio = prio;
    al.t_ms = t_ms;
    ps_phrase_t parts[3] = { p0, p1, p2 };
    for (int i = 0; i < 3; i++) if (parts[i] != PS_PHRASE_NONE) al.parts[al.n_parts++] = parts[i];
    if (!al.n_parts) return;

    /* Drop an identical message already waiting (no stacking repeats). */
    for (int i = 0; i < a->n; i++)
        if (a->q[i].n_parts == al.n_parts && !memcmp(a->q[i].parts, al.parts, sizeof(al.parts))) {
            a->q[i].t_ms = t_ms;
            return;
        }

    if (a->n < PS_ALERT_QUEUE) { a->q[a->n++] = al; return; }
    int low = 0;
    for (int i = 1; i < a->n; i++) if (a->q[i].prio < a->q[low].prio) low = i;
    if (a->q[low].prio < prio) a->q[low] = al;
    else a->dropped++;
}

bool ps_alerts_pop(ps_alerts_t *a, ps_alert_t *out)
{
    if (!a->n) return false;
    int best = 0;
    for (int i = 1; i < a->n; i++)
        if (a->q[i].prio > a->q[best].prio ||
            (a->q[i].prio == a->q[best].prio && (int32_t)(a->q[i].t_ms - a->q[best].t_ms) < 0))
            best = i;
    *out = a->q[best];
    for (int i = best; i + 1 < a->n; i++) a->q[i] = a->q[i + 1];
    a->n--;
    return true;
}

int ps_alerts_peek_max_prio(const ps_alerts_t *a)
{
    int m = -1;
    for (int i = 0; i < a->n; i++) if ((int)a->q[i].prio > m) m = (int)a->q[i].prio;
    return m;
}

ps_phrase_t ps_direction_phrase(float b)
{
    static const ps_phrase_t sector[8] = {
        PS_PHRASE_DIR_AHEAD, PS_PHRASE_DIR_AHEAD_LEFT, PS_PHRASE_DIR_LEFT, PS_PHRASE_DIR_BEHIND_LEFT,
        PS_PHRASE_DIR_BEHIND, PS_PHRASE_DIR_BEHIND_RIGHT, PS_PHRASE_DIR_RIGHT, PS_PHRASE_DIR_AHEAD_RIGHT,
    };
    int s = (int)lroundf(b / 45.0f);
    s = ((s % 8) + 8) % 8;
    return sector[s];
}

void ps_alerts_update_nav(ps_alerts_t *a, const ps_config_t *cfg,
                          const ps_nav_guidance_t *g, uint32_t t_ms)
{
    if (!g->valid) return;

    /* Hysteresis so the level does not chatter around a threshold. */
    const float hyst = 0.05f;
    ps_navconf_level_t lvl = a->level;
    if (g->confidence < cfg->nav_conf_unreliable) lvl = PS_NAVCONF_UNRELIABLE;
    else if (g->confidence < cfg->nav_conf_warn) {
        if (lvl == PS_NAVCONF_GOOD || g->confidence >= cfg->nav_conf_unreliable + hyst)
            lvl = PS_NAVCONF_DEGRADED;
    } else if (g->confidence >= cfg->nav_conf_warn + hyst) lvl = PS_NAVCONF_GOOD;

    const ps_phrase_t dir = ps_direction_phrase(g->route_bearing_rel_deg);

    if (lvl != a->level) {
        if (lvl == PS_NAVCONF_DEGRADED) {
            /* Confidence dropping: say the way out while it is still usable. */
            ps_alerts_push(a, PS_PRIO_NAV, t_ms, PS_PHRASE_WAY_OUT_IS, dir, PS_PHRASE_NONE);
            a->t_last_direction_ms = t_ms;
        } else if (lvl == PS_NAVCONF_UNRELIABLE) {
            ps_alerts_push(a, PS_PRIO_WARNING, t_ms, PS_PHRASE_NAV_UNRELIABLE, PS_PHRASE_FOLLOW_HOSE, PS_PHRASE_NONE);
            a->t_last_unreliable_ms = t_ms;
        } else if (a->level == PS_NAVCONF_UNRELIABLE) {
            ps_alerts_push(a, PS_PRIO_INFO, t_ms, PS_PHRASE_NAV_RECOVERED, PS_PHRASE_NONE, PS_PHRASE_NONE);
        }
        a->level = lvl;
    } else if (lvl == PS_NAVCONF_DEGRADED && t_ms - a->t_last_direction_ms >= cfg->direction_repeat_ms) {
        ps_alerts_push(a, PS_PRIO_NAV, t_ms, PS_PHRASE_WAY_OUT_IS, dir, PS_PHRASE_NONE);
        a->t_last_direction_ms = t_ms;
    } else if (lvl == PS_NAVCONF_UNRELIABLE && t_ms - a->t_last_unreliable_ms >= cfg->unreliable_repeat_ms) {
        ps_alerts_push(a, PS_PRIO_WARNING, t_ms, PS_PHRASE_NAV_UNRELIABLE, PS_PHRASE_FOLLOW_HOSE, PS_PHRASE_NONE);
        a->t_last_unreliable_ms = t_ms;
    }

    /* Arrived back at the door after having gone somewhere. */
    if (g->home_dist_m > 4.0f) {
        a->left_entry = true;
    } else if (a->left_entry && g->home_dist_m < 1.5f) {
        ps_alerts_push(a, PS_PRIO_INFO, t_ms, PS_PHRASE_AT_EXIT, PS_PHRASE_NONE, PS_PHRASE_NONE);
        a->left_entry = false;
    }
}

void ps_alerts_request_direction(ps_alerts_t *a, const ps_nav_guidance_t *g, uint32_t t_ms)
{
    if (!g->valid) return;
    ps_phrase_t dir = ps_direction_phrase(g->route_bearing_rel_deg);
    if (a->level == PS_NAVCONF_UNRELIABLE)
        ps_alerts_push(a, PS_PRIO_WARNING, t_ms, PS_PHRASE_NAV_UNRELIABLE, PS_PHRASE_FOLLOW_HOSE, PS_PHRASE_NONE);
    else
        ps_alerts_push(a, PS_PRIO_NAV, t_ms, PS_PHRASE_WAY_OUT_IS, dir, PS_PHRASE_NONE);
    a->t_last_direction_ms = t_ms;
}

void ps_alerts_update_detections(ps_alerts_t *a, const ps_config_t *cfg,
                                 const ps_detections_t *dets, uint32_t t_ms)
{
    /* Spoken person call-outs only from the neural detector: the threshold
     * fallback is too easily fooled by warm surfaces to justify interrupting
     * the wearer. Its boxes are still drawn. */
    if (dets->source != PS_DETECTOR_NEURAL) return;
    const ps_detection_t *best = NULL;
    for (int i = 0; i < dets->n; i++)
        if (dets->d[i].cls == PS_CLASS_PERSON && dets->d[i].score >= 0.5f &&
            (!best || dets->d[i].score > best->score))
            best = &dets->d[i];
    if (!best) return;

    bool after_gap = !a->person_seen || t_ms - a->t_last_person_seen_ms > cfg->person_alert_cooldown_ms;
    bool cooled = !a->person_seen || t_ms - a->t_last_person_alert_ms >= cfg->person_alert_cooldown_ms;
    a->person_seen = true;
    a->t_last_person_seen_ms = t_ms;
    if (!after_gap || !cooled) return;

    /* Horizontal position in the image -> ahead / left / right. */
    const float hfov = cfg->hfov_deg;
    float cx = best->x + best->w * 0.5f;
    float rel = (0.5f - cx / PS_THERM_W) * hfov; /* + = left of centre */
    ps_phrase_t dir = fabsf(rel) < hfov / 6 ? PS_PHRASE_DIR_AHEAD
                    : (rel > 0 ? PS_PHRASE_DIR_AHEAD_LEFT : PS_PHRASE_DIR_AHEAD_RIGHT);
    ps_alerts_push(a, PS_PRIO_WARNING, t_ms, PS_PHRASE_PERSON, dir, PS_PHRASE_NONE);
    a->t_last_person_alert_ms = t_ms;
}
