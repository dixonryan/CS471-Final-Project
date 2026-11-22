#!/usr/bin/env python3
"""
hmm_predict.py -- predict where an object will be, using a Hidden Markov Model.
Plain Python, no libraries.  Notation follows CS471 Lesson 23.

HOW TO RUN (one argument, nothing else):
    python hmm_predict.py test_track.csv

  The CSV needs a header row t,x,y,z.  t is in seconds; gaps can be uneven.

  OUTPUT
    CURRENT ESTIMATE  one line per report: where the HMM thinks the object
                      really is at that moment (reports are noisy, so this is
                      an estimate, not a copy of the report), and which way it
                      is most likely heading.
    PREDICTIONS       where the object will be 2, 5 and 10 seconds after the
                      last report, with an uncertainty (+/-).

THE HMM
  Hidden state X_t : (box, direction)
                       box       = one box of a 3D grid of equal-size boxes
                                   that goes on forever in every direction
                       direction = +x, -x, +y, -y, +z, -z, or STOPPED
  Evidence e_t     : a noisy (x, y, z) location report
  Transition model : P(X_t | X_t-1)  -- direction table below, then move one box
  Emission model   : P(e_t | X_t)    -- reports land near the true box
  Prior            : B(X_0)          -- every state near the first report equally likely

FORWARD ALGORITHM (filtering), once per report
  Predict:  B_bar(X_t) = sum over x_t-1 of  P(X_t | x_t-1) * B(x_t-1)
  Update :  B(X_t)     = alpha * P(e_t | X_t) * B_bar(X_t)
            alpha      = 1 / (sum of P(e_t | x_t) * B_bar(x_t))

PREDICTING THE FUTURE
  After the last report, keep doing the Predict step (no Update, because there
  are no more reports) until we reach the time we want.

HOW THE BELIEF IS STORED
  B is a Python dictionary:   B[(box, direction)] = probability
  e.g.  B[((3, 4, 1), "+x")] = 0.02  means "2% chance the object is in box
  (3, 4, 1) heading +x".  All the probabilities in B add up to 1.
  The grid is endless, so B only holds states with a real chance of being
  true; any state not in B has probability 0.
"""

import csv        # reads CSV files
import math       # sqrt, exp, floor
import sys        # reads the command-line argument

# ---------------------------------------------------------------------------
# SETTINGS -- change these here (there are no command-line options)
# ---------------------------------------------------------------------------
PREDICT_TIMES = [2, 5, 10]   # seconds after the last report to predict
SIGMA = 0.7                  # sensor noise: about how far off a report is (same units as x,y,z)
BOX_SIZE = 1.0               # width of one grid box (same units as x,y,z)
MIN_PROB = 1e-6              # states less likely than this are dropped (keeps it fast)

# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------
# The 7 possible directions.  Together with a box, a direction makes one
# hidden state.
DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z", "STOPPED"]

# How many boxes each direction moves in (x, y, z) per step.
STEP = {"+x": (1, 0, 0), "-x": (-1, 0, 0),
        "+y": (0, 1, 0), "-y": (0, -1, 0),
        "+z": (0, 0, 1), "-z": (0, 0, -1),
        "STOPPED": (0, 0, 0)}

# The reverse of each moving direction (for the "turn completely around" probability).
OPPOSITE = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y", "+z": "-z", "-z": "+z"}

# ---------------------------------------------------------------------------
# Transition table: P(next direction | current direction), per box traveled
# ---------------------------------------------------------------------------
P_KEEP = 0.90       # keep going the same way
P_TURN = 0.02       # each 90-degree turn (there are 4 of them)
P_REVERSE = 0.01    # turn completely around
P_STOP = 0.01       # stop
# check: 0.90 + 4(0.02) + 0.01 + 0.01 = 1.00

P_STAY_STOPPED = 0.90                     # a stopped object stays stopped...
P_START = (1 - P_STAY_STOPPED) / 6        # ...or starts moving in any of the 6 directions


def direction_change_prob(old, new):
    """P(new direction | old direction) -- looks up the transition table."""
    if old == "STOPPED":
        # stopped objects either stay stopped or start moving
        return P_STAY_STOPPED if new == "STOPPED" else P_START
    if new == old:                  # same direction
        return P_KEEP
    if new == "STOPPED":            # moving -> stopped
        return P_STOP
    if new == OPPOSITE[old]:        # 180-degree turn
        return P_REVERSE
    return P_TURN                   # anything else is a 90-degree turn


# ---------------------------------------------------------------------------
# The grid (endless, so these are just conversions between boxes and points)
# ---------------------------------------------------------------------------
def box_of(point):
    """Which box an (x, y, z) point falls in.  Box (0, 0, 0) covers 0 to BOX_SIZE
    on each axis; box (-1, 0, 0) is the one just below it in x, and so on."""
    return tuple(math.floor(point[i] / BOX_SIZE) for i in range(3))


def center(box):
    """Real-world (x, y, z) at the center of a box."""
    return tuple((box[i] + 0.5) * BOX_SIZE for i in range(3))


def move(box, direction):
    """The box one step over in `direction`."""
    step = STEP[direction]
    return tuple(box[i] + step[i] for i in range(3))


# ---------------------------------------------------------------------------
# Emission model:  P(e_t | X_t)
# ---------------------------------------------------------------------------
def emission_prob(report, box):
    """The closer the report is to the box, the more likely it is.
    (Gaussian sensor noise.  It isn't scaled to sum to 1, but alpha takes care
    of that in the Update step.)"""
    c = center(box)
    # squared distance from the report to the center of the box
    dist_squared = sum((report[i] - c[i]) ** 2 for i in range(3))
    # Gaussian: 1.0 when the report is exactly on the box, shrinking toward 0
    # as the report gets farther away.  SIGMA sets how fast it shrinks.
    return math.exp(-dist_squared / (2 * SIGMA ** 2))


# ---------------------------------------------------------------------------
# Forward algorithm
# ---------------------------------------------------------------------------
def drop_tiny(B):
    """Remove states with almost no chance of being true, so B stays small."""
    return {state: p for state, p in B.items() if p >= MIN_PROB}


def prior(report):
    """B(X_0): before using any reports, every state is equally likely.
    The grid is endless, so we only list the boxes within 3 sigma of the first
    report -- every box farther away gets ~0 probability as soon as the first
    report is applied, so leaving them out changes nothing."""
    middle = box_of(report)
    r = math.ceil(3 * SIGMA / BOX_SIZE)         # how many boxes is 3 sigma
    B = {}
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                box = (middle[0] + dx, middle[1] + dy, middle[2] + dz)
                for d in DIRECTIONS:
                    B[(box, d)] = 1.0
    n = len(B)
    return {state: 1 / n for state in B}        # equal probability, adds up to 1


def predict_one_box(B):
    """Predict step for one box of travel:
        B_bar(X_t) = sum over x_t-1 of P(X_t | x_t-1) * B(x_t-1)
    From each old state, the object picks a new direction (transition table)
    and moves one box that way."""
    B_bar = {}
    # go through every old state and its probability
    for (box, d), p in B.items():
        # the object could switch to any of the 7 directions...
        for new_d in DIRECTIONS:
            p_transition = direction_change_prob(d, new_d)     # P(new dir | old dir)
            new_state = (move(box, new_d), new_d)               # ...and moves one box that way
            # add this path's share of probability to the state it ends up in
            B_bar[new_state] = B_bar.get(new_state, 0.0) + p_transition * p
    return drop_tiny(B_bar)


def predict(B, speed, dt):
    """Predict step for a time gap dt.  The object travels speed * dt, which is
    some number of boxes, e.g. 2.4.  Do 2 full one-box steps, then a 0.4 step:
    40% chance it made the extra move, 60% chance it hasn't yet."""
    boxes_traveled = speed * dt / BOX_SIZE      # distance in boxes, e.g. 2.4
    whole = int(boxes_traveled)                 # whole boxes, e.g. 2
    leftover = boxes_traveled - whole           # fraction left, e.g. 0.4
    # one predict step per whole box traveled
    for _ in range(whole):
        B = predict_one_box(B)
    # for the fraction: mix "moved one more box" and "didn't move yet"
    if leftover > 0:
        moved = predict_one_box(B)
        mixed = {}
        for state, p in B.items():
            mixed[state] = mixed.get(state, 0.0) + (1 - leftover) * p
        for state, p in moved.items():
            mixed[state] = mixed.get(state, 0.0) + leftover * p
        B = mixed
    return B


def update(B_bar, report):
    """Update step:
        B(X_t) = alpha * P(e_t | X_t) * B_bar(X_t)"""
    # multiply each state's predicted probability by how well it matches the report
    B = {(box, d): emission_prob(report, box) * p for (box, d), p in B_bar.items()}
    total = sum(B.values())
    if total == 0:
        # the report is nowhere near anything we predicted -> start over from it
        return update(prior(report), report)
    # alpha = 1 / total, so the new belief adds up to 1
    alpha = 1 / total
    B = {state: alpha * p for state, p in B.items()}
    return drop_tiny(B)


def run_filter(times, reports, speed, show=False):
    """Forward algorithm over all reports.  Returns the final belief B.
    If show=True, prints the belief after each report."""
    # first report: start from the prior and update with it
    B = update(prior(reports[0]), reports[0])
    if show:
        print_filter_header()
        print_filter_row(times[0], reports[0], summarize(B))
    # every later report: predict forward by the time gap, then update
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]                # time since last report
        B_bar = predict(B, speed, dt)               # Predict step
        B = update(B_bar, reports[i])               # Update step
        if show:
            print_filter_row(times[i], reports[i], summarize(B))
    return B


def predict_future(B, speed):
    """Run the Predict step (no Update) out to each time in PREDICT_TIMES.
    Goes in order -- e.g. predict 2 s, then 3 s more to reach 5 s, then 5 s
    more to reach 10 s -- so the work isn't repeated.
    Returns {time: belief at that time}."""
    beliefs = {}
    t_done = 0
    for t in sorted(PREDICT_TIMES):
        B = predict(B, speed, t - t_done)
        beliefs[t] = B
        t_done = t
    return beliefs


# ---------------------------------------------------------------------------
# Reading the answer out of the belief
# ---------------------------------------------------------------------------
def summarize(B):
    """Turn a belief B into readable answers: expected position, uncertainty,
    and most likely direction."""
    total = sum(B.values())     # ~1 (a little less after dropping tiny states)
    box_prob = {}               # P(box): add up the 7 directions in each box
    dir_prob = {d: 0.0 for d in DIRECTIONS}     # P(direction): add up all boxes
    for (box, d), p in B.items():
        box_prob[box] = box_prob.get(box, 0.0) + p / total
        dir_prob[d] += p / total

    # expected position = sum of P(box) * center of box  (a weighted average)
    mean = [0.0, 0.0, 0.0]
    for box, p in box_prob.items():
        c = center(box)
        for i in range(3):
            mean[i] += p * c[i]

    # uncertainty = standard deviation of distance from the expected position
    var = 0.0
    for box, p in box_prob.items():
        c = center(box)
        var += p * sum((c[i] - mean[i]) ** 2 for i in range(3))

    best_dir = max(dir_prob, key=dir_prob.get)  # direction with highest probability
    return {"mean": tuple(mean), "std": math.sqrt(var),
            "dir": best_dir, "dir_prob": dir_prob[best_dir]}


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------
def estimate_speed(times, reports, gap=3):
    """Speed = distance / time between reports `gap` apart; take the median so
    a few noisy reports don't throw it off.  Distance is |dx| + |dy| + |dz|
    because the object only moves along one axis at a time in this model
    (a diagonal path is a mix of, e.g., +x and +y moves)."""
    gap = min(gap, len(times) - 1)      # can't use a gap bigger than the track
    speeds = []
    # compare report i with report i + gap, for every i
    for i in range(len(times) - gap):
        dt = times[i + gap] - times[i]
        if dt > 0:
            dist = sum(abs(reports[i + gap][k] - reports[i][k]) for k in range(3))
            speeds.append(dist / dt)
    if not speeds:
        return 0.0
    # median = middle value after sorting
    speeds.sort()
    return speeds[len(speeds) // 2]


# ---------------------------------------------------------------------------
# Input / output
# ---------------------------------------------------------------------------
def read_track(path):
    """Read t,x,y,z rows; returns (times, points) sorted by time."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):           # each row is {"t": ..., "x": ..., ...}
            rows.append((float(row["t"]), float(row["x"]), float(row["y"]), float(row["z"])))
    rows.sort()                                 # sorts by t (the first item)
    times = [r[0] for r in rows]                # list of times
    points = [r[1:] for r in rows]              # list of (x, y, z)
    return times, points


def fmt(p):
    """Format an (x, y, z) point for printing."""
    return "(" + ", ".join(f"{v:7.2f}" for v in p) + ")"


def print_filter_header():
    """Column titles for the CURRENT ESTIMATE table."""
    print("CURRENT ESTIMATE (after each report)")
    print(f"{'t':>8}  {'report':>27}  {'estimated position':>27}  {'+/-':>5}  "
          f"{'heading':>8} {'P':>5}")


def print_filter_row(t, report, s):
    """One row of the CURRENT ESTIMATE table (s is a summarize() result):
    time, the raw report, the HMM's estimated position, its uncertainty, and
    the most likely heading with its probability."""
    print(f"{t:8.2f}  {fmt(report)}  {fmt(s['mean'])}  {s['std']:5.2f}  "
          f"{s['dir']:>8} {s['dir_prob']:5.2f}")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    """Read the CSV, run the HMM over every report, then predict the future."""
    # exactly one argument: the CSV file
    if len(sys.argv) != 2:
        print("Usage:  python hmm_predict.py <track.csv>")
        return
    times, reports = read_track(sys.argv[1])        # load the reports
    speed = estimate_speed(times, reports)          # one speed for the whole track
    # summary line: how much data, and the settings being used
    print(f"{len(times)} reports | estimated speed {speed:.2f} | box size {BOX_SIZE} "
          f"| sensor noise {SIGMA}\n")

    # run the forward algorithm over every report -> belief at the last report
    B = run_filter(times, reports, speed, show=True)

    # Predict step only (no Update) out to each time in PREDICT_TIMES
    print("\n\nPREDICTIONS")              # blank lines separate it from the table above
    print(f"{'ahead':>8}  {'t':>8}  {'predicted position':>27}  {'+/-':>6}")
    for t, B_future in predict_future(B, speed).items():
        s = summarize(B_future)                     # expected position + uncertainty
        # seconds ahead, the clock time it refers to, the prediction, +/- uncertainty
        print(f"{t:>7g}s  {times[-1] + t:8.2f}  {fmt(s['mean'])}  {s['std']:6.2f}")


# Only run main() when this file is run directly (not when imported)
if __name__ == "__main__":
    main()
