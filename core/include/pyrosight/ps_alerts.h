/*
 * PyroSight core: audio alert logic.
 *
 * The core decides WHAT to say and WHEN; the firmware's audio component plays
 * pre-recorded clips (one per phrase id) through the codec to the 3.5 mm jack.
 * Messages are built from short clips, e.g.
 *   [WAY_OUT_IS] [DIR_BEHIND_LEFT]
 *   [NAV_UNRELIABLE] [FOLLOW_HOSE]
 * so no speech synthesis is needed on the device.
 */
#ifndef PS_ALERTS_H
#define PS_ALERTS_H

#include "ps_config.h"
#include "ps_detect.h"
#include "ps_nav.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    PS_PHRASE_NONE = 0,
    PS_PHRASE_WAY_OUT_IS,        /* "Way out is" */
    PS_PHRASE_DIR_AHEAD,         /* "ahead" */
    PS_PHRASE_DIR_AHEAD_LEFT,    /* "ahead, to your left" */
    PS_PHRASE_DIR_LEFT,          /* "to your left" */
    PS_PHRASE_DIR_BEHIND_LEFT,   /* "behind you, to the left" */
    PS_PHRASE_DIR_BEHIND,        /* "behind you" */
    PS_PHRASE_DIR_BEHIND_RIGHT,  /* "behind you, to the right" */
    PS_PHRASE_DIR_RIGHT,         /* "to your right" */
    PS_PHRASE_DIR_AHEAD_RIGHT,   /* "ahead, to your right" */
    PS_PHRASE_NAV_UNRELIABLE,    /* "Navigation estimate unreliable." */
    PS_PHRASE_FOLLOW_HOSE,       /* "Follow the hose line out." */
    PS_PHRASE_NAV_RECOVERED,     /* "Navigation restored." */
    PS_PHRASE_PERSON,            /* "Person" */
    PS_PHRASE_FIRE,              /* "Fire" */
    PS_PHRASE_CAMERA_LOST,       /* "Thermal camera lost." */
    PS_PHRASE_CAMERA_RESTORED,   /* "Thermal camera restored." */
    PS_PHRASE_MOTION_LOST,       /* "Motion sensor lost." */
    PS_PHRASE_ENTRY_MARKED,      /* "Entry point marked." */
    PS_PHRASE_AT_EXIT,           /* "You are at the entry point." */
    PS_PHRASE_BATTERY_LOW,       /* "Battery low." */
    PS_PHRASE_BATTERY_CRITICAL,  /* "Battery critical. Exit now." */
    PS_PHRASE_COUNT
} ps_phrase_t;

/* English text of each phrase, used to generate clips (tools/make_audio_clips.py). */
extern const char *const ps_phrase_text[PS_PHRASE_COUNT];

typedef enum {
    PS_PRIO_INFO = 0,
    PS_PRIO_NAV = 1,
    PS_PRIO_WARNING = 2,
    PS_PRIO_CRITICAL = 3,
} ps_alert_prio_t;

#define PS_ALERT_MAX_PARTS 4
#define PS_ALERT_QUEUE 8

typedef struct {
    ps_phrase_t parts[PS_ALERT_MAX_PARTS];
    uint8_t n_parts;
    ps_alert_prio_t prio;
    uint32_t t_ms;
} ps_alert_t;

typedef enum {
    PS_NAVCONF_GOOD = 0,
    PS_NAVCONF_DEGRADED,
    PS_NAVCONF_UNRELIABLE,
} ps_navconf_level_t;

typedef struct {
    ps_alert_t q[PS_ALERT_QUEUE];
    uint8_t n;
    ps_navconf_level_t level;
    uint32_t t_last_direction_ms;
    uint32_t t_last_unreliable_ms;
    uint32_t t_last_person_seen_ms;
    uint32_t t_last_person_alert_ms;
    bool left_entry;        /* has been > 4 m from the door since the last arrival */
    bool person_seen;
    uint32_t dropped;
} ps_alerts_t;

void ps_alerts_init(ps_alerts_t *a);

/* Push an alert; a higher-priority alert evicts the lowest queued one if full. */
void ps_alerts_push(ps_alerts_t *a, ps_alert_prio_t prio, uint32_t t_ms,
                    ps_phrase_t p0, ps_phrase_t p1, ps_phrase_t p2);

/* Pop the highest-priority (oldest first within priority) alert. */
bool ps_alerts_pop(ps_alerts_t *a, ps_alert_t *out);

/* Highest priority currently queued, or -1 if the queue is empty. Lets the
 * audio player cut a message short when something more urgent arrives. */
int ps_alerts_peek_max_prio(const ps_alerts_t *a);

/* Relative bearing (+ = left) to one of the eight direction phrases. */
ps_phrase_t ps_direction_phrase(float rel_bearing_deg);

/* Navigation-driven alerts; call every tick. */
void ps_alerts_update_nav(ps_alerts_t *a, const ps_config_t *cfg,
                          const ps_nav_guidance_t *g, uint32_t t_ms);

/* The wearer pressed "where is out?". Always answers. */
void ps_alerts_request_direction(ps_alerts_t *a, const ps_nav_guidance_t *g, uint32_t t_ms);

/* Detection-driven alerts (a person appearing after a gap). */
void ps_alerts_update_detections(ps_alerts_t *a, const ps_config_t *cfg,
                                 const ps_detections_t *dets, uint32_t t_ms);

#ifdef __cplusplus
}
#endif

#endif /* PS_ALERTS_H */
