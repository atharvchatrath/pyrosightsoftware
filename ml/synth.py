"""Synthetic thermal scene generator for the PyroSight detector.

Produces raw radiometric frames as the Lepton 3.5 would deliver them in
low-gain radiometric mode: int16 deci-degrees C, 160x120, *before* the
device's denoise stage. Labels are tight boxes around the visible part of
each fire / person.

This is a domain-randomised toy renderer, not a physical simulation. It is good
enough to exercise the training/eval/export pipeline and to give the network
sensible priors (fires are very hot and irregular, people are ~33-36 C heads on
cooler bodies with a known size-vs-distance relation), but it will not
generalise to real fireground imagery on its own. See ml/README.md.

Scene recipe (all temperatures in C until the final deci-C conversion):
  * room: ambient 15..60 C, floor slightly cooler, ceiling gas layer that heats
    up towards the top of the frame (up to several hundred C in fire scenes);
  * structure: walls with vertical seams, door frames (sometimes a hot room
    behind), furniture blocks at near-ambient temperature (also used as
    occluders in front of people/fires);
  * fires 200..600 C: irregular flickering blobs with flame tongues, a warm
    plume above, partly occluded;
  * people standing / crouching / crawling / lying at 1..12 m, sized with the
    pinhole model f_px = 80 / tan(28.5 deg) and ~1.7 m body length; skin
    33..36 C, clothing or turnout gear between skin and ambient, head hottest;
  * hard negatives: hot pipes, radiators, lamps (50..120 C), sunlit windows,
    hot walls; people that are cooler than a hot background;
  * smoke 0..1: mild attenuation toward the smoke temperature (more with
    distance), blur, extra noise;
  * sensor: optics blur, temporal noise 0.3..0.5 C (+smoke), column
    fixed-pattern noise, a few dead / stuck pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

W, H = 160, 120
HFOV_DEG = 57.0
F_PX = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))   # ~147.4 px
BODY_LEN_M = 1.7
CLS_FIRE, CLS_PERSON = 0, 1

_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)


@dataclass
class Sample:
    dc: np.ndarray                        # int16 (H, W) deci-C, raw (not denoised)
    boxes: np.ndarray                     # float32 (N, 4) x, y, w, h (top-left, px)
    cls: np.ndarray                       # int64 (N,)
    dist: np.ndarray                      # float32 (N,) metres (0 for fire)
    smoke: float = 0.0
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers
def gauss_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.15:
        return img
    k = gauss_kernel(sigma)
    r = len(k) // 2
    p = np.pad(img, ((0, 0), (r, r)), mode="edge")
    out = sum(k[i] * p[:, i:i + img.shape[1]] for i in range(len(k)))
    p = np.pad(out, ((r, r), (0, 0)), mode="edge")
    return sum(k[i] * p[i:i + img.shape[0], :] for i in range(len(k))).astype(np.float32)


def smooth_noise(rng, cell: float, shape=(H, W)) -> np.ndarray:
    """Value noise in ~[-1, 1] with feature size `cell` px (bilinear upsampled)."""
    gh = int(math.ceil(shape[0] / cell)) + 2
    gw = int(math.ceil(shape[1] / cell)) + 2
    g = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
    ys = np.arange(shape[0], dtype=np.float32) / cell
    xs = np.arange(shape[1], dtype=np.float32) / cell
    y0 = ys.astype(int); x0 = xs.astype(int)
    fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
    a = g[y0][:, x0]; b = g[y0][:, x0 + 1]; c = g[y0 + 1][:, x0]; d = g[y0 + 1][:, x0 + 1]
    fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def paint(img, cov, temp):
    """Alpha-blend `temp` (scalar or array) into img with coverage cov in [0,1]."""
    np.multiply(img, 1 - cov, out=img)
    img += cov * temp


def rect_cov(x0, y0, x1, y1):
    """Anti-aliased coverage of an axis-aligned rectangle [x0,x1) x [y0,y1)."""
    cx = np.clip(np.minimum(_XX + 1, x1) - np.maximum(_XX, x0), 0, 1)
    cy = np.clip(np.minimum(_YY + 1, y1) - np.maximum(_YY, y0), 0, 1)
    return (cx * cy).astype(np.float32)


def capsule_cov(p0, p1, r):
    """Anti-aliased coverage of a capsule (thick segment) of radius r px."""
    px, py = _XX + 0.5, _YY + 0.5
    ax, ay = p0; bx, by = p1
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
    d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
    return np.clip(r - d + 0.5, 0, 1).astype(np.float32)


def ellipse_cov(cx, cy, rx, ry):
    d = np.hypot((_XX + 0.5 - cx) / max(rx, 1e-3), (_YY + 0.5 - cy) / max(ry, 1e-3))
    return np.clip((1 - d) * max(min(rx, ry), 0.5) + 0.5, 0, 1).astype(np.float32)


def box_of(mask, thr=0.5):
    ys, xs = np.nonzero(mask > thr)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()], np.float32)


# ---------------------------------------------------------------- people
# Skeletons in body-length units: x right, y up is negative, feet/ground at y=0.
# Each part: (kind, p0, p1, radius, role) ; role in {"head", "body", "limb"}.
def _skeleton(pose, rng):
    j = lambda s=0.015: rng.normal(0, s)
    if pose == "standing":
        arm_sw = rng.uniform(-0.08, 0.08)
        leg_sp = rng.uniform(0.02, 0.09)
        return [
            ("cap", (-0.05, -0.47), (-leg_sp + j(), 0.0), 0.055, "body"),
            ("cap", (0.05, -0.47), (leg_sp + j(), 0.0), 0.055, "body"),
            ("cap", (0.0, -0.80), (0.0, -0.48), 0.11, "body"),
            ("cap", (-0.14, -0.79), (-0.16 + arm_sw, -0.45 + j()), 0.04, "body"),
            ("cap", (0.14, -0.79), (0.16 - arm_sw, -0.45 + j()), 0.04, "body"),
            ("ell", (0.0, -0.925), None, 0.065, "head"),
        ]
    if pose == "crouching":
        return [
            ("cap", (-0.06, -0.30), (-0.10, -0.02), 0.06, "body"),
            ("cap", (0.06, -0.30), (0.14, -0.02), 0.06, "body"),
            ("cap", (0.0, -0.50), (0.02 + j(), -0.28), 0.11, "body"),
            ("cap", (-0.12, -0.50), (-0.05 + j(), -0.28), 0.04, "body"),
            ("cap", (0.12, -0.50), (0.18 + j(), -0.30), 0.04, "body"),
            ("ell", (0.03 + j(), -0.61), None, 0.065, "head"),
        ]
    if pose == "crawling":   # hands and knees, side view, facing +x
        return [
            ("cap", (-0.26, -0.36), (0.18, -0.40 + j()), 0.10, "body"),
            ("cap", (-0.26, -0.33), (-0.24, -0.04), 0.06, "body"),
            ("cap", (-0.24, -0.04), (-0.55, -0.03), 0.05, "body"),
            ("cap", (0.16, -0.38), (0.20 + j(), -0.02), 0.04, "body"),
            ("ell", (0.31, -0.45 + j()), None, 0.065, "head"),
        ]
    # lying, side view
    return [
        ("cap", (-0.30, -0.09), (0.12, -0.10 + j(0.01)), 0.10, "body"),
        ("cap", (-0.30, -0.07), (-0.76, -0.05 + j(0.01)), 0.06, "body"),
        ("cap", (0.05, -0.12), (0.18 + j(0.03), -0.04), 0.04, "body"),
        ("ell", (0.24, -0.11), None, 0.065, "head"),
    ]


def render_person(rng, feet_x, feet_y, L_px, pose, mirror):
    """Returns (body_cov, head_cov) coverage maps for one person."""
    body = np.zeros((H, W), np.float32)
    head = np.zeros((H, W), np.float32)
    s = -1.0 if mirror else 1.0
    for kind, p0, p1, r, role in _skeleton(pose, rng):
        P0 = (feet_x + s * p0[0] * L_px, feet_y + p0[1] * L_px)
        if kind == "cap":
            P1 = (feet_x + s * p1[0] * L_px, feet_y + p1[1] * L_px)
            c = capsule_cov(P0, P1, max(r * L_px, 0.45))
        else:
            c = ellipse_cov(P0[0], P0[1], max(r * L_px * 0.85, 0.5), max(r * L_px, 0.5))
        tgt = head if role == "head" else body
        np.maximum(tgt, c, out=tgt)
    return body, head


# ---------------------------------------------------------------- fire
def render_fire(rng, cx, base_y, w_px, h_px):
    """Irregular flickering flame. Returns (coverage, temperature map C)."""
    nx = (_XX + 0.5 - cx) / max(w_px / 2, 0.5)
    ny = (base_y - (_YY + 0.5)) / max(h_px, 0.5)           # 0 at base, 1 at top
    turb = smooth_noise(rng, max(1.5, w_px / 4)) * 0.35 + smooth_noise(rng, max(1.0, w_px / 9)) * 0.2
    # Flame narrows upward; tongues from the noise field.
    width = np.clip(1 - np.clip(ny, 0, None) ** 1.3, 0, 1) * (1 + turb)
    inside = (np.abs(nx + 0.25 * turb * ny) < width) & (ny > -0.08) & (ny < 1 + 0.4 * turb)
    cov = blur(inside.astype(np.float32), 0.6)
    peak = rng.uniform(200, 600)
    core = np.clip(1 - np.abs(nx) * 0.8 - ny * 0.6 + 0.3 * turb, 0.15, 1)
    flick = 1 + 0.25 * smooth_noise(rng, max(1.0, w_px / 6))
    temp = 110 + (peak - 110) * core * flick
    return np.clip(cov, 0, 1), np.clip(temp, 90, 650).astype(np.float32)


# ---------------------------------------------------------------- scene
def generate(seed: int, p_empty: float = 0.12) -> Sample:
    rng = np.random.default_rng(seed)
    ambient = float(rng.uniform(15, 35) if rng.random() < 0.6 else rng.uniform(35, 60))
    smoke = float(np.clip(rng.choice([0.0, rng.uniform(0, 0.2), rng.uniform(0.2, 0.5), rng.uniform(0.5, 1.0)],
                                     p=[0.25, 0.2, 0.3, 0.25]), 0, 1))
    n_fire = int(rng.choice([0, 1, 2, 3], p=[0.45, 0.35, 0.15, 0.05]))
    n_person = int(rng.choice([0, 1, 2, 3, 4], p=[0.3, 0.35, 0.2, 0.1, 0.05]))
    if rng.random() < p_empty:
        n_fire = n_person = 0
    fire_scene = n_fire > 0 or rng.random() < 0.3

    cam_h = float(rng.choice([rng.uniform(1.3, 1.75), rng.uniform(0.4, 0.9)], p=[0.6, 0.4]))  # standing / crawling FF
    horizon = H / 2 + rng.uniform(-18, 18)
    tilt = rng.uniform(-0.05, 0.05)           # roll, px per px
    hz = horizon + tilt * (_XX - W / 2)

    # --- background: ceiling gas layer, walls, floor
    layer_dT = rng.uniform(80, 400) if fire_scene and rng.random() < 0.7 else rng.uniform(0, 25)
    layer_bottom = horizon - rng.uniform(-10, 45)        # rows above this get hotter
    t_up = np.clip((layer_bottom - _YY) / max(layer_bottom, 8), 0, 1)
    img = (ambient + layer_dT * t_up ** rng.uniform(0.8, 2.0)).astype(np.float32)
    img += 1.5 * smooth_noise(rng, rng.uniform(10, 40))
    floor = np.clip((_YY - hz) / 12, 0, 1)
    paint(img, floor, ambient - rng.uniform(0, 4) + 1.0 * smooth_noise(rng, 20))

    # vertical wall seams / corners
    for _ in range(rng.integers(0, 4)):
        x = rng.uniform(0, W)
        paint(img, rect_cov(x, 0, x + rng.uniform(1, 6), H) * rng.uniform(0.3, 1), img + rng.uniform(-4, 4))

    smoke_t = ambient + 0.4 * layer_dT * 0.3 + rng.uniform(0, 10)
    labels = []
    occluders = []          # (cov, temp) drawn after objects

    def atten(dist_m):      # fraction of an object's contrast lost to smoke
        return float(np.clip(smoke * (0.05 + 0.025 * dist_m) * rng.uniform(0.7, 1.3), 0, 0.5))

    def ground_row(dist_m, x):
        return float(horizon + tilt * (x - W / 2) + F_PX * cam_h / dist_m)

    # door frames (sometimes with a hot room behind)
    for _ in range(rng.integers(0, 3)):
        d = rng.uniform(2, 12)
        dw, dh = F_PX * 0.9 / d, F_PX * 2.05 / d
        x0 = rng.uniform(-dw / 2, W - dw / 2)
        y1 = ground_row(d, x0 + dw / 2)
        behind = ambient + (rng.uniform(20, 250) if fire_scene and rng.random() < 0.4 else rng.uniform(-8, 8))
        paint(img, rect_cov(x0, y1 - dh, x0 + dw, y1), behind + 3 * smooth_noise(rng, 8))
        fr = max(0.8, F_PX * 0.08 / d)
        ft = ambient + rng.uniform(-3, 5)
        for cov in (rect_cov(x0 - fr, y1 - dh - fr, x0, y1), rect_cov(x0 + dw, y1 - dh - fr, x0 + dw + fr, y1),
                    rect_cov(x0 - fr, y1 - dh - fr, x0 + dw + fr, y1 - dh)):
            paint(img, cov, ft)

    # windows: sunlit glass / bright exterior (person-like temperatures: hard negative)
    if rng.random() < 0.25:
        d = rng.uniform(3, 12)
        ww, wh_ = F_PX * rng.uniform(0.6, 1.5) / d, F_PX * rng.uniform(0.6, 1.2) / d
        x0 = rng.uniform(-ww / 3, W - ww / 2)
        y0 = ground_row(d, x0) - F_PX * rng.uniform(1.0, 2.0) / d
        paint(img, rect_cov(x0, y0, x0 + ww, y0 + wh_), rng.uniform(30, 55) + 2 * smooth_noise(rng, 6))

    # hard negatives: pipes, radiators, lamps
    for _ in range(rng.integers(0, 5) if rng.random() < 0.6 else 0):
        kind = rng.choice(["pipe", "radiator", "lamp"])
        T = rng.uniform(50, 120)
        d = rng.uniform(1.5, 12)
        if kind == "pipe":
            r = max(0.6, F_PX * rng.uniform(0.02, 0.06) / d)
            if rng.random() < 0.5:
                y = rng.uniform(0, H); p0, p1 = (rng.uniform(-40, 80), y), (rng.uniform(80, 200), y + rng.uniform(-5, 5))
            else:
                x = rng.uniform(0, W); p0, p1 = (x, rng.uniform(-20, 60)), (x + rng.uniform(-5, 5), rng.uniform(60, 140))
            paint(img, capsule_cov(p0, p1, r), T)
        elif kind == "radiator":
            rw, rh = F_PX * rng.uniform(0.5, 1.2) / d, F_PX * rng.uniform(0.4, 0.7) / d
            x0 = rng.uniform(-rw / 2, W - rw / 2); y1 = ground_row(d, x0) - F_PX * 0.1 / d
            fins = 0.85 + 0.15 * np.cos(_XX * 2 * np.pi / max(2.0, rw / rng.integers(4, 12)))
            paint(img, rect_cov(x0, y1 - rh, x0 + rw, y1), ambient + (T - ambient) * fins)
        else:
            r = max(0.7, F_PX * rng.uniform(0.04, 0.12) / d)
            cx, cy = rng.uniform(0, W), rng.uniform(0, horizon)
            paint(img, ellipse_cov(cx, cy, r, r * rng.uniform(0.7, 1.4)), T + rng.uniform(0, 60))

    # objects sorted far -> near so nearer ones occlude farther ones
    objs = []
    for _ in range(n_fire):
        objs.append(("fire", float(rng.uniform(1.5, 15))))
    for _ in range(n_person):
        objs.append(("person", float(np.clip(rng.choice([rng.uniform(1.0, 12.0), rng.uniform(2.0, 7.0)]), 1.0, 12.0))))
    objs.sort(key=lambda o: -o[1])

    obj_masks = []
    for kind, d in objs:
        a = atten(d)
        if kind == "fire":
            fw = F_PX * rng.uniform(0.25, 1.6) / d
            fh = fw * rng.uniform(0.7, 2.2)
            cx = rng.uniform(-fw * 0.2, W + fw * 0.2)
            base = ground_row(d, cx) - (F_PX * rng.uniform(0, 0.8) / d if rng.random() < 0.3 else 0)
            cov, temp = render_fire(rng, cx, base, fw, fh)
            if cov.max() < 0.5:
                continue
            # plume above the flame (warm, unlabelled)
            plume = blur(cov, max(1.0, fw / 3)) * np.clip((base - fh * 0.6 - _YY) / max(fh, 1), 0, 1)
            paint(img, np.clip(plume * 1.5, 0, 0.8), ambient + 0.25 * (temp - ambient))
            temp = temp + (smoke_t - temp) * a
            paint(img, cov, temp)
            obj_masks.append([CLS_FIRE, cov, 0.0])
        else:
            pose = str(rng.choice(["standing", "crouching", "crawling", "lying"], p=[0.4, 0.2, 0.2, 0.2]))
            L = F_PX * BODY_LEN_M * rng.uniform(0.9, 1.1) / d
            fx = rng.uniform(-0.1 * W, 1.1 * W)
            fy = ground_row(d, fx) + rng.normal(0, 1)
            body, head = render_person(rng, fx, fy, L, pose, bool(rng.random() < 0.5))
            if max(body.max(), head.max()) < 0.5:
                continue
            skin = rng.uniform(33, 36)
            gear = rng.random() < 0.6           # turnout gear: much closer to ambient
            alpha = rng.uniform(0.3, 0.7) if gear else rng.uniform(0.15, 0.45)
            cloth = skin + (ambient - skin) * alpha + rng.uniform(-1.5, 1.5)
            if ambient > 45 and gear:
                cloth = min(cloth, ambient - rng.uniform(2, 8))   # cooler than a hot room
            if abs(cloth - ambient) < 2.0:      # keep a minimum (pre-smoke) contrast of ~2-4 C
                sign = 1.0 if (skin > ambient and not (ambient > 45 and gear)) else -1.0
                cloth = ambient + sign * rng.uniform(2.0, 4.0)
            tex = 0.8 * smooth_noise(rng, max(1.0, L / 12))
            paint(img, body, cloth + tex + (smoke_t - cloth) * a)
            head_t = skin + 0.5 * rng.normal()
            if gear and rng.random() < 0.4:      # helmet/hood: only the face (lower head) is hot
                face = head * np.clip((_YY + 0.5 - (fy - L * (0.93 if pose == "standing" else 0.6))) /
                                      max(L * 0.05, 0.5) + 0.3, 0, 1)
                paint(img, head, cloth + 1.0)
                head = face
            paint(img, head, head_t + (smoke_t - head_t) * a)
            obj_masks.append([CLS_PERSON, np.maximum(body, head), d])

        # furniture / debris occluding the most recent object (partial occlusion)
        if rng.random() < 0.3 and obj_masks:
            b = box_of(obj_masks[-1][1])
            if b is not None:
                ox0 = b[0] + rng.uniform(-0.3, 0.7) * b[2]
                ow = b[2] * rng.uniform(0.3, 0.8)
                oy0 = b[1] + b[3] * rng.uniform(0.55, 0.8)
                occ = rect_cov(ox0, oy0, ox0 + ow, H + 5)
                paint(img, occ, ambient + rng.uniform(-3, 3) + smooth_noise(rng, 6))
                for m in obj_masks:
                    m[1] = m[1] * (1 - occ)
        # nearer objects occlude earlier masks too
        if len(obj_masks) >= 2:
            newest = obj_masks[-1][1]
            for m in obj_masks[:-1]:
                m[1] = m[1] * (1 - newest)

    # foreground furniture blocks not tied to an object
    for _ in range(rng.integers(0, 3)):
        d = rng.uniform(1.0, 6)
        fw, fh = F_PX * rng.uniform(0.4, 1.5) / d, F_PX * rng.uniform(0.3, 0.9) / d
        x0 = rng.uniform(-fw / 2, W - fw / 2)
        y1 = ground_row(d, x0 + fw / 2)
        if y1 - fh > H:
            continue
        occ = rect_cov(x0, y1 - fh, x0 + fw, y1)
        paint(img, occ, ambient + rng.uniform(-4, 6) + smooth_noise(rng, 5))
        for m in obj_masks:
            m[1] = m[1] * (1 - occ)

    # --- smoke: global veil + blur + extra noise
    if smoke > 0:
        veil = np.clip(smoke * rng.uniform(0.05, 0.2) * (1 + 0.5 * smooth_noise(rng, 30)), 0, 0.4)
        paint(img, veil, smoke_t)
    img = blur(img, rng.uniform(0.3, 0.7) + 1.1 * smoke * rng.uniform(0.5, 1.0))

    # --- sensor
    sigma = rng.uniform(0.3, 0.5) + smoke * rng.uniform(0.2, 0.8)
    img += rng.normal(0, sigma, img.shape).astype(np.float32)
    img += rng.normal(0, rng.uniform(0.1, 0.45), (1, W)).astype(np.float32)       # column FPN
    img += rng.normal(0, 0.1, (H, 1)).astype(np.float32)                           # weak row FPN
    img += rng.normal(0, 1.0)                                                      # radiometric offset
    dc = np.clip(np.round(img * 10), -400, 6500).astype(np.int16)
    for _ in range(rng.integers(0, 6)):
        y, x = rng.integers(0, H), rng.integers(0, W)
        dc[y, x] = rng.choice([-400, 0, 6500, int(dc[y, x]) + int(rng.integers(-300, 300))])

    # --- labels from visible masks
    boxes, clss, dists = [], [], []
    for c, m, d in obj_masks:
        b = box_of(m, 0.5)
        if b is None:
            continue
        min_side = 2 if c == CLS_FIRE else 3
        if b[2] < min_side or b[3] < min_side or (m > 0.5).sum() < (4 if c == CLS_FIRE else 6):
            continue
        boxes.append(b); clss.append(c); dists.append(d)
    return Sample(
        dc=dc,
        boxes=np.array(boxes, np.float32).reshape(-1, 4),
        cls=np.array(clss, np.int64),
        dist=np.array(dists, np.float32),
        smoke=round(smoke, 2),
        meta=dict(ambient=ambient, layer_dT=layer_dT, cam_h=cam_h),
    )


if __name__ == "__main__":
    import time
    t = time.time()
    for i in range(50):
        s = generate(i)
    print(f"{(time.time() - t) / 50 * 1e3:.1f} ms/frame; last: {len(s.cls)} objects, smoke {s.smoke}")
