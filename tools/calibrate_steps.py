#!/usr/bin/env python3
"""Per-wearer navigation calibration from field trials.

Walk a measured course several times in full gear (and, ideally, in the
crouched "duck walk" used in smoke) and record the device's step count for
each run. Optionally spin in place a known number of turns and record the
yaw change the device reported. Put the runs in a CSV:

    kind,known,measured
    walk,20.0,33        # walked 20.0 m, device counted 33 steps
    walk,20.0,35
    crouch,20.0,41
    turn,1080,1071      # turned 3 full turns, device yaw changed 1071 deg

Prints the values to put in ps_config_t (core/src/ps_config.c or the
firmware's NVS config): step_length_m, step_length_bias, and the gyro turn
scale error to compare against the heading-uncertainty model.
"""
import csv
import statistics
import sys


def main(path):
    walks, crouch, turns = [], [], []
    with open(path) as f:
        for row in csv.DictReader(line for line in f if line.strip() and not line.startswith("#")):
            kind = row["kind"].strip()
            known, meas = float(row["known"]), float(row["measured"].split("#")[0])
            if kind == "walk":
                walks.append(known / meas)
            elif kind == "crouch":
                crouch.append(known / meas)
            elif kind == "turn":
                turns.append(meas / known - 1.0)
    if not walks:
        sys.exit("need at least one 'walk' row")
    L = statistics.mean(walks)
    spread = statistics.pstdev(walks) / L if len(walks) > 1 else 0.05
    # Systematic error: gait changes between upright and crouched walking are
    # the dominant bias in smoke, so fold that difference in when measured.
    bias = spread
    if crouch:
        bias = max(bias, abs(statistics.mean(crouch) - L) / L)
    print(f"runs: {len(walks)} upright, {len(crouch)} crouched, {len(turns)} turn tests")
    print(f"step_length_m      = {L:.3f}")
    print(f"step_length_sigma  = {max(spread, 0.03):.3f}   (run-to-run spread)")
    print(f"step_length_bias   = {max(bias, 0.03):.3f}   (systematic, incl. crouched gait)")
    if crouch:
        print(f"crouched stride    = {statistics.mean(crouch):.3f} m ({100 * (statistics.mean(crouch) / L - 1):+.0f}% vs upright)")
    if turns:
        s = statistics.mean(abs(t) for t in turns)
        print(f"gyro turn scale    = {100 * s:.2f}% (model assumes 1.0% per turn; raise ps_nav.c turn term if higher)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
