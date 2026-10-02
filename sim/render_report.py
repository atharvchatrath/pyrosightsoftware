#!/usr/bin/env python3
"""Turn a simulator run (ps_sim --out DIR) into pictures:
  DIR/eyepiece.gif   the composited eyepiece view, as the wearer sees it
  DIR/track.png      floor plan with true path vs dead-reckoned estimate

    python3 sim/render_report.py DIR
"""
import csv
import glob
import os
import sys

from PIL import Image, ImageDraw

# Same floor plan as sim/ps_sim.c (door at origin, facing +x).
WALLS = [(0, -1, 15, -1), (0, 1, 13, 1), (15, -1, 15, 12), (13, 1, 13, 12), (9, 12, 13, 12),
         (15, 12, 19, 12), (19, 12, 19, 20), (9, 20, 19, 20), (9, 12, 9, 20)]
FIRE, CASUALTY, CREW = (17.6, 18.8), (10.6, 16.5), (14.0, 9.5)


def gif(d):
    frames = [Image.open(f).convert("P", palette=Image.ADAPTIVE)
              for f in sorted(glob.glob(os.path.join(d, "frames", "*.ppm")))]
    if frames:
        frames[0].save(os.path.join(d, "eyepiece.gif"), save_all=True, append_images=frames[1:],
                       duration=230, loop=0, optimize=True)


def track(d):
    rows = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
    S, ox, oy, W, H = 26, 40, 40, 20 * 26 + 80, 22 * 26 + 80
    img = Image.new("RGB", (W, H), (250, 250, 248))
    dr = ImageDraw.Draw(img)
    P = lambda x, y: (ox + x * S, H - oy - (y + 1) * S)  # world (y up) to image
    for x0, y0, x1, y1 in WALLS:
        dr.line([P(x0, y0), P(x1, y1)], fill=(40, 40, 40), width=4)
    for (x, y), col, lab in ((FIRE, (230, 80, 0), "fire"), (CASUALTY, (0, 150, 200), "casualty"),
                             (CREW, (0, 150, 200), "crew")):
        cx, cy = P(x, y)
        dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=col)
        dr.text((cx + 10, cy - 6), lab, fill=col)
    dx, dy = P(0, 0)
    dr.rectangle([dx - 6, dy - 6, dx + 6, dy + 6], outline=(0, 140, 0), width=3)
    dr.text((dx + 8, dy + 8), "door", fill=(0, 140, 0))
    true = [P(float(r["true_x"]), float(r["true_y"])) for r in rows]
    est = [P(float(r["est_x"]), float(r["est_y"])) for r in rows]
    dr.line(true, fill=(30, 30, 30), width=2)
    dr.line(est, fill=(200, 40, 160), width=2)
    dr.text((10, 10), "black: true path   magenta: device estimate (entry frame)", fill=(0, 0, 0))
    summ = os.path.join(d, "summary.txt")
    if os.path.exists(summ):
        for i, line in enumerate(l for l in open(summ) if l.startswith(("navigation", "way out"))):
            dr.text((10, 26 + 14 * i), line.strip(), fill=(0, 0, 0))
    img.save(os.path.join(d, "track.png"))


if __name__ == "__main__":
    d = sys.argv[1]
    gif(d)
    track(d)
    print("wrote", os.path.join(d, "eyepiece.gif"), "and", os.path.join(d, "track.png"))
