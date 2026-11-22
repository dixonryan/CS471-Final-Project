#!/usr/bin/env python3
"""
generate_tracks.py -- make synthetic test tracks for hmm_predict.py.
Plain Python, no libraries.

HOW TO RUN (no arguments):
    python generate_tracks.py

It writes:
  test_track.csv       one sample track: noisy location reports at uneven times
  test_track_key.csv   its answer key (see below)
  tests/               50 more tracks: test_track_001.csv ... test_track_050.csv
  test_keys/           their answer keys: test_track_001_key.csv ...

Every run makes exactly the same files (fixed random seeds).

ANSWER KEYS
  hmm_predict.py never reads these.  They hold the object's TRUE position at
  the same times the program predicts (2, 5 and 10 s after the last report),
  so predictions can be checked by hand or by an AI:

      seconds_after_last_report,t,x,y,z
      2,59.680,12.540,-36.960,22.920
      5,...
      10,...

THE SIMULATED OBJECT
  Moves through open space (no walls) at a random constant speed between 0.5
  and 1.5 units/s.  Its heading slowly wanders, so the path gently curves.
  Reports come 0.5 to 3 s apart and have Gaussian noise with std SIGMA.
"""

import csv        # writes CSV files
import math       # sqrt
import os         # makes the folders
import random     # random numbers for the simulation

# ---------------------------------------------------------------------------
# SETTINGS -- change these here (there are no command-line options)
# ---------------------------------------------------------------------------
REPORTS_PER_TRACK = 30   # location reports in each track
SIGMA = 0.7              # sensor noise (should match SIGMA in hmm_predict.py)
KEY_TIMES = [2, 5, 10]   # seconds after the last report in each answer key
                         # (should match PREDICT_TIMES in hmm_predict.py)
NUM_TESTS = 50           # tracks in the tests folder
SAMPLE_SEED = 3          # random seed for test_track.csv
TESTS_SEED = 0           # random seed for the tests folder


def unit(v):
    """Scale a vector to length 1 (keeps its direction)."""
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def interpolate(times, points, t):
    """Position on the path at time t (straight line between saved points)."""
    for i in range(1, len(times)):
        if times[i] >= t:                                   # found the points around t
            f = (t - times[i - 1]) / (times[i] - times[i - 1])  # how far between them (0 to 1)
            return tuple(points[i - 1][k] + f * (points[i][k] - points[i - 1][k]) for k in range(3))
    return points[-1]                                       # t is past the end


def simulate_track(rng):
    """Simulate one object.  Returns (report_times, reports, key_rows) where
    key_rows are (seconds after last report, t, true x, true y, true z)."""
    dt = 0.1                                    # simulation time step (seconds)

    # random starting point somewhere in a 20 x 20 x 10 region (just a place
    # to start -- the object is free to go anywhere after that)
    pos = [rng.uniform(0, 20), rng.uniform(0, 20), rng.uniform(0, 10)]
    # random starting heading (a random direction of length 1)
    heading = unit([rng.gauss(0, 1) for _ in range(3)])
    # random speed, constant for the whole track
    speed = rng.uniform(0.5, 1.5)

    # report times: 0.5 to 3 seconds apart (uneven on purpose)
    report_times = [0.0]
    for _ in range(REPORTS_PER_TRACK - 1):
        report_times.append(report_times[-1] + rng.uniform(0.5, 3.0))
    last = report_times[-1]
    # keep simulating past the last report, far enough for the answer key
    end = last + max(KEY_TIMES) + 1

    # move the object in small time steps and record the true path
    t = 0.0
    truth_times, truth_points = [0.0], [tuple(pos)]
    while t < end:
        # nudge the heading a little at random, so the path slowly curves
        heading = unit([heading[i] + rng.gauss(0, 0.15 * math.sqrt(dt)) for i in range(3)])
        for i in range(3):
            pos[i] += speed * heading[i] * dt              # move forward
        t += dt
        truth_times.append(t)
        truth_points.append(tuple(pos))

    # reports = true position at each report time + Gaussian noise (std = SIGMA)
    reports = []
    for rt in report_times:
        true_p = interpolate(truth_times, truth_points, rt)
        reports.append(tuple(true_p[i] + rng.gauss(0, SIGMA) for i in range(3)))

    # answer key: the true position at each KEY_TIMES after the last report
    key_rows = []
    for k in KEY_TIMES:
        p = interpolate(truth_times, truth_points, last + k)
        key_rows.append((k, last + k, p[0], p[1], p[2]))
    return report_times, reports, key_rows


def write_track(path, times, points):
    """Write the noisy reports as a t,x,y,z CSV file."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z"])                   # header row
        for t, p in zip(times, points):
            w.writerow([f"{t:.3f}", f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}"])


def write_key(path, key_rows):
    """Write the answer key CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seconds_after_last_report", "t", "x", "y", "z"])
        for k, t, x, y, z in key_rows:
            w.writerow([k, f"{t:.3f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])


def save_test_case(track_path, key_path, rng):
    """Simulate one track and save its reports and its answer key."""
    times, reports, key_rows = simulate_track(rng)
    write_track(track_path, times, reports)
    write_key(key_path, key_rows)


def main():
    # the sample track (same seed -> same track every time)
    save_test_case("test_track.csv", "test_track_key.csv", random.Random(SAMPLE_SEED))
    print("Wrote test_track.csv (noisy reports) and test_track_key.csv (answer key)")

    # 50 test tracks in tests/, their answer keys in test_keys/
    os.makedirs("tests", exist_ok=True)                    # make the folders if needed
    os.makedirs("test_keys", exist_ok=True)
    rng = random.Random(TESTS_SEED)
    for k in range(NUM_TESTS):
        name = f"test_track_{k + 1:03d}"                   # test_track_001, _002, ...
        save_test_case(os.path.join("tests", name + ".csv"),
                       os.path.join("test_keys", name + "_key.csv"), rng)
    print(f"Wrote {NUM_TESTS} test tracks to tests/ and their answer keys to test_keys/")


# Only run main() when this file is run directly (not when imported)
if __name__ == "__main__":
    main()
