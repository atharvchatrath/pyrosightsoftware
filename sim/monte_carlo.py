#!/usr/bin/env python3
"""Run every simulator scenario over many seeds and summarise navigation and
detection results (the "navigation accuracy" and "edge case" test bench).

    python3 sim/monte_carlo.py [--seeds 20] [--sim build/sim/ps_sim] [--out results.md]
"""
import argparse
import re
import statistics
import subprocess

SCENARIOS = ["corridor", "long", "crawl", "dropout"]


def run(sim, scenario, seed, extra):
    out = subprocess.run([sim, "--scenario", scenario, "--seed", str(seed), "--quiet", *extra],
                         check=True, capture_output=True, text=True).stdout
    nav = re.search(r"max position error ([\d.]+) m, lowest confidence ([\d.]+)", out)
    way = re.search(r"way out\s+(.*); walk-out took ([\d.]+) s; ended ([\d.]+) m from the door", out)
    person = re.search(r"detector person\s+precision ([\d.]+) recall ([\d.]+)", out)
    fire = re.search(r"detector fire\s+precision ([\d.]+) recall ([\d.]+)", out)
    outcome = way.group(1)
    end_err = float(way.group(3))
    if "following the arrow" in outcome or ("device said EXIT" in outcome and end_err <= 1.5):
        kind = "arrow"
    elif "hose" in outcome and "NOT" not in outcome:
        kind = "hose"
    else:
        kind = "fail"
    return {
        "max_err": float(nav.group(1)), "min_conf": float(nav.group(2)), "kind": kind,
        "end_err": end_err, "walk_s": float(way.group(2)),
        "person_p": float(person.group(1)), "person_r": float(person.group(2)),
        "fire_p": float(fire.group(1)), "fire_r": float(fire.group(2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--sim", default="build/sim/ps_sim")
    ap.add_argument("--out")
    ap.add_argument("--no-snap", action="store_true")
    a = ap.parse_args()
    extra = ["--no-snap"] if a.no_snap else []
    lines = ["| scenario | out by arrow | out by hose | failed | median max nav error (m) | worst (m) | fire P/R | person P/R (threshold detector) |",
             "|---|---|---|---|---|---|---|---|"]
    for sc in SCENARIOS:
        rs = [run(a.sim, sc, s, extra) for s in range(1, a.seeds + 1)]
        n = len(rs)
        cnt = {k: sum(r["kind"] == k for r in rs) for k in ("arrow", "hose", "fail")}
        errs = [r["max_err"] for r in rs]
        mean = lambda k: statistics.mean(r[k] for r in rs)
        lines.append(f"| {sc} | {cnt['arrow']}/{n} | {cnt['hose']}/{n} | {cnt['fail']}/{n} | "
                     f"{statistics.median(errs):.2f} | {max(errs):.2f} | {mean('fire_p'):.2f}/{mean('fire_r'):.2f} | "
                     f"{mean('person_p'):.2f}/{mean('person_r'):.2f} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
