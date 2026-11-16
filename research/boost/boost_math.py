"""Star Fox (2026) boost math: a frame-by-frame simulation of the boost meter
and the boost speed of the Arwing, the Landmaster and the Blue-Marine, at 60 fps.

The values were datamined from the game files (each vehicle's boost script and
the player vehicle's own settings) and checked frame by frame against the game
running its own code.

Run:  python boost_math.py
It prints the tables of the write-up. With matplotlib installed
(pip install matplotlib) it also draws the charts as PNG files.
"""

import random
import struct

FPS = 60


def f32(x):
    """Round to a 32-bit float, as the game computes."""
    return struct.unpack("f", struct.pack("f", x))[0]


DT = f32(1 / FPS)  # one frame, in seconds

# Boost speed is the extra forward speed on top of your normal speed.
# Rates are per second; at 60 fps a frame is 1/60 of that.
ARWING = dict(name="Arwing", meter_max=90, fill=60, drain=30, accel=1800, max_boost=900, decel=900, brake=900, brake_floor=600, brake_recover=450)
BLUE_MARINE = dict(name="Blue-Marine", meter_max=90, fill=40, drain=20, accel=250, max_boost=300, decel=100, brake=125, brake_floor=300, brake_recover=100)
LANDMASTER = dict(name="Landmaster", meter_max=90, rate=30, boost_target=300, brake_target=-300)


class Linear:
    """Arwing and Blue-Marine. Boosting (or braking) fills the meter; the
    moment you stop, the meter drains, and boost and brake stay locked until it
    is completely empty. Boost speed rises at a fixed rate up to its max while
    boosting, and falls at a fixed rate when not."""

    def __init__(self, v):
        self.v = v
        self.fill, self.drain = f32(v["fill"] * DT), f32(v["drain"] * DT)
        self.up, self.down = f32(v["accel"] * DT), f32(v["decel"] * DT)
        self.meter, self.speed, self.locked = 0.0, 0.0, False

    def frame(self, boost_held):
        if boost_held and not self.locked:
            self.meter = f32(self.meter + self.fill)
            if self.meter > self.v["meter_max"]:
                self.meter, self.locked = float(self.v["meter_max"]), True
            self.speed = min(f32(self.speed + self.up), self.v["max_boost"])
            return True
        if self.meter > 0:
            self.meter = f32(self.meter - self.drain)
            if self.meter <= 0:
                self.meter, self.locked = 0.0, False
            else:
                self.locked = True
        self.speed = max(f32(self.speed - self.down), 0.0)
        return False


def add_calc(value, target, rate, cap, minimum):
    """The game's easing function: 30 times a step toward the target, `rate`
    of the gap but at most `cap` either way; a step smaller than `minimum`
    becomes +minimum (even when the target is below); 0 at the target."""
    if value == target:
        return 0.0
    step = f32(rate * f32(target - value))
    step = max(-cap, min(cap, step)) if abs(step) >= minimum else minimum
    return f32(30 * step)


class Landmaster:
    """The Landmaster's boost speed eases toward 300 (fast at first, capped
    at 3 per frame, then slower and slower) and fades by 5% per frame when not
    boosting. The meter fills and drains at the same rate; when you let go
    before the meter is full, the lock starts one frame before it drains."""

    def __init__(self, v):
        self.v = v
        self.rate = f32(v["rate"] * DT)
        self.meter, self.speed, self.locked = 0.0, 0.0, False

    def ease(self, target, share, cap):
        self.speed = f32(self.speed + f32(DT * add_calc(self.speed, target, f32(share), cap, f32(0.001))))

    def frame(self, boost_held):
        boosting = boost_held and not self.locked
        if boosting:
            self.ease(self.v["boost_target"], 0.2, 6.0)
            self.meter = f32(self.meter + self.rate)
            if self.meter > self.v["meter_max"]:
                self.meter, self.locked = float(self.v["meter_max"]), True
        else:
            self.ease(0.0, 0.1, 100.0)
            if self.meter > 0:
                if not self.locked:
                    self.locked = True  # the release frame: no drain yet
                else:
                    self.meter = f32(self.meter - self.rate)
                    if self.meter <= 1.1920929e-4:
                        self.meter, self.locked = 0.0, False
        return boosting


def make(v):
    return Landmaster(v) if v is LANDMASTER else Linear(v)


def boost_for(n):
    """Hold boost for n frames, then let go until boost is available again."""
    state = {"left": n}

    def held(vehicle):
        if vehicle.locked or vehicle.meter > 0 and state["left"] == 0:
            state["left"] = n if not vehicle.locked and vehicle.meter == 0 else 0
            return False
        if state["left"] == 0:
            state["left"] = n
        state["left"] -= 1
        return True

    return held


def hold_always(vehicle):
    return True


def hold_and_release(period, release):
    """Keep boost held, letting go for `release` frames every `period` frames."""
    count = {"i": -1}

    def held(vehicle):
        count["i"] += 1
        return count["i"] % period >= release

    return held


def until_speed(limit):
    """Hold boost, and let go for one frame each time a boost frame leaves the boost
    speed at `limit` or more."""
    return lambda vehicle: vehicle.locked or vehicle.meter == 0 or vehicle.speed < limit


def exact_rule(start=(68, 45, 34), rest=8):
    """The Blue-Marine's rule played exactly, with boost held all along: the first boosts
    last `start` frames (1/2, 1/3, then 1/4 of the meter), every later one `rest` frames
    (a flick about 0.1 s after each restart, played without timing errors); each boost
    restarts by itself the moment its cooldown ends."""
    state = {"boost": 0, "frames": 0}

    def held(vehicle):
        length = start[state["boost"]] if state["boost"] < len(start) else rest
        if state["frames"] >= length:
            state["boost"], state["frames"] = state["boost"] + 1, 0
            return False
        if not vehicle.locked:
            state["frames"] += 1
        return True

    return held


class Player:
    """A human who holds boost and flicks it off for `flick` frames. Timing errors are
    about `sigma` frames, a reaction takes `reaction` frames (12 = 0.2 s, give or take 2),
    and each button change reaches the game 1 or 2 frames late. The first boosts are let
    go when the meter shows `start` frames of boost (68, 45, 34: 1/2, 1/3, 1/4 of the
    Blue-Marine's meter), never sooner than a reaction. Then, by `mode`:
      "react":      let go when you see the boost restart;
      "anticipate": let go `aim` frames after the boost should restart (twice the last
                    boost after it ended); when a flick misses, react;
      "rhythm":     let go every `every` frames, without looking."""

    def __init__(self, mode, start=(68, 45, 34), every=12, aim=6, sigma=2.0, flick=3, reaction=12, seed=0):
        self.mode, self.start, self.every, self.aim = mode, start, every, aim
        self.sigma, self.flick, self.reaction = sigma, flick, reaction
        self.rng, self.flicks = random.Random(seed), []  # flicks: (first, end) frames, as the game reads them
        self.t, self.meter, self.boosting, self.began, self.boosts, self.beat = -1, 0.0, False, 0, 0, None

    def let_go(self, at):
        self.flicks.append((at + self.rng.randint(1, 2), at + self.flick + self.rng.randint(1, 2)))

    def react(self):
        return min(max(round(self.rng.gauss(self.reaction, 2)), self.reaction - 3), self.reaction + 3)

    def __call__(self, vehicle):
        self.t += 1
        t, boosted = self.t, vehicle.meter > self.meter  # did the last frame boost?
        self.meter = vehicle.meter
        if boosted and not self.boosting:  # a boost started on the last frame
            k, self.began = self.boosts, t - 1
            self.boosts += 1
            if k < len(self.start):
                self.let_go(t - 1 + max(round(self.rng.gauss(self.start[k], self.sigma)), self.react()))
            elif self.mode == "react" or self.mode == "anticipate" and all(lo < t for lo, _ in self.flicks):
                self.let_go(t - 1 + self.react())  # (anticipating: the planned flick missed)
            elif self.mode == "rhythm" and self.beat is None:
                self.beat = t + self.rng.randrange(self.every)
        elif self.boosting and not boosted and self.mode == "anticipate" and self.boosts >= len(self.start):
            boost = t - 1 - self.began  # a boost just ended: the cooldown lasts twice as long
            self.let_go(t - 1 + 2 * boost + round(self.rng.gauss(self.aim, self.sigma)))
        self.boosting = boosted
        if self.beat is not None and t >= self.beat:
            self.let_go(self.beat)
            self.beat += max(round(self.rng.gauss(self.every, self.sigma)), self.flick + 2)
        while self.flicks and self.flicks[0][1] <= t:
            self.flicks.pop(0)
        return not any(lo <= t < end for lo, end in self.flicks)


def run(v, pattern, seconds=60):
    """Frame by frame: (meter, boost speed, boosting) of each frame."""
    vehicle, out = make(v), []
    for _ in range(seconds * FPS):
        boosting = vehicle.frame(pattern(vehicle))
        out.append((vehicle.meter, vehicle.speed, boosting))
    return out


def average(trace):
    """Average boost speed over the whole cycles of the second half, once the
    cycles have settled (a cycle starts with the first frame of a boost)."""
    half = len(trace) // 2
    starts = [i for i in range(half, len(trace)) if trace[i][2] and not trace[i - 1][2]]
    lo, hi = (starts[0], starts[-1]) if len(starts) >= 2 else (half, len(trace))
    return sum(s for _, s, _ in trace[lo:hi]) / (hi - lo)


def full_meter_frames(v):
    return round(v["meter_max"] / (v.get("fill", v.get("rate")) / FPS))


def curve(v, pattern, seconds):
    """Boost speed over time, starting from a standstill boost-wise."""
    trace = run(v, pattern, seconds)
    return [i / FPS for i in range(len(trace))], [s for _, s, _ in trace]


AQUAS = 102_200  # units from the first boost to the boss arena of Aquas (3:53 speedrun video)


def aquas_time(pattern):
    """Seconds from the first boost to the boss arena of Aquas, starting with no boost
    speed. Each frame moves the Blue-Marine (cruise speed 400 + boost speed) / 60 units."""
    vehicle, units, frames = make(BLUE_MARINE), 0.0, 0
    while units < AQUAS:
        vehicle.frame(pattern(vehicle))
        units += (400 + vehicle.speed) / FPS
        frames += 1
    return frames / FPS


def aquas_human(mode, runs=16, **player):
    """Average Aquas time of `runs` players (see Player)."""
    return sum(aquas_time(Player(mode, seed=seed, **player)) for seed in range(runs)) / runs


def chart(plt, v, file, seconds, lines, max_frames, marks):
    """Left: boost speed over time for a few ways to boost. Right: the average
    boost speed for each boost length, as the share of the meter used."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2))
    for label, pattern in lines:
        t, s = curve(v, pattern, seconds)
        left.plot(t, s, label=label, linewidth=1.0)
    left.set(title=f"{v['name']}: boost speed over time", xlabel="time (s)", ylabel="boost speed (extra speed)")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(lines), fontsize=9, frameon=False)
    left.grid(alpha=0.3)
    full = full_meter_frames(v)
    ns = list(range(1, max_frames + 1))
    avg = [average(run(v, boost_for(n))) for n in ns]
    pct = [min(n, full) / full * 100 for n in ns]
    right.plot(pct, avg, color="black", linewidth=1.6)
    for n, text in marks:
        i = n - 1
        right.plot(pct[i], avg[i], "o", color="crimson")
        high = (avg[i] - min(avg)) / (max(avg) - min(avg)) > 0.5
        offset = ((-62, -34) if high else (-62, 8)) if pct[i] > 85 else (8, -28)
        right.annotate(f"{text}\n{avg[i]:.0f}", (pct[i], avg[i]), textcoords="offset points", xytext=offset, fontsize=9)
    right.set(title=f"{v['name']}: average boost speed vs when you let go", xlabel="meter used before letting go (%)", ylabel="average boost speed")
    right.set_xlim(0, 100)
    right.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(file, dpi=100)
    plt.close(fig)
    print(f"wrote {file}")


def aquas_chart(plt, file="bluemarine-aquas.png"):
    """Aquas time for every boost length, frame-perfect and by hand, and the best rules."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    full = full_meter_frames(BLUE_MARINE)
    ns, hand = range(1, full + 1), [1, *range(5, full + 1, 5)]
    ax.plot([n / full * 100 for n in ns], [aquas_time(boost_for(n)) for n in ns], color="black", linewidth=1.6, label="frame-perfect, same boost length every time")
    times = [aquas_human("react", runs=4, start=(n,) * 999) for n in hand]  # let go at n frames every time
    ax.plot([n / full * 100 for n in hand], times, color="crimson", linewidth=1.2, label="by hand, aiming at the same boost length every time")
    for y, text in [
        (aquas_time(until_speed(296)), "frame-perfect: let go once boost speed is 296"),
        (aquas_human("anticipate"), "by hand: long start, then flick 0.1 s after each restart"),
        (164.6, "the 3:53 speedrun, measured on its video"),
    ]:
        ax.axhline(y, linestyle="--", linewidth=1.0, color="gray")
        ax.annotate(f"{text}: {y:.1f} s", (100, y), textcoords="offset points", xytext=(-4, 4), ha="right", fontsize=9)
    ax.set(title="Blue-Marine on Aquas: first boost to the boss arena", xlabel="meter used before letting go (%)", ylabel="time (s)")
    ax.set(xlim=(0, 100), ylim=(145, 192))
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(file, dpi=100)
    plt.close(fig)
    print(f"wrote {file}")


def levels(v):
    """Ways to boost, side by side: the best a robot can play, the best rule a human can
    play (played exactly), and letting go at a fixed share of the meter. Each entry is
    (label, a function giving a fresh pattern, line style)."""
    full = full_meter_frames(v)

    def share(p, label):
        n = max(1, round(full * p))
        return (f"let go at {label} of the meter ({n} frames)", lambda: boost_for(n), "-")

    robot, human = {
        "Arwing": (("perfect robot: let go at 1/3 (30 frames)", lambda: boost_for(30)), ("perfect human: the same, 1/3", lambda: boost_for(30))),
        "Blue-Marine": (("perfect robot: let go once boost speed is 296", lambda: until_speed(296)), ("perfect human: 1/2, 1/3, 1/4, then 8-frame flicks", lambda: exact_rule())),
        "Landmaster": (("perfect robot: hold it (100%)", lambda: hold_always), ("perfect human: the same, hold it", lambda: hold_always)),
    }[v["name"]]
    return [
        (*robot, "-"),
        (*human, "--"),
        ("hold until the meter is full (100%)", lambda: hold_always, "-"),
        share(1 / 2, "50%"),
        share(1 / 3, "33%"),
        share(1 / 4, "25%"),
        share(1 / 5, "20%"),
    ]


def levels_chart(plt, v, file, seconds):
    """Left: boost speed over the first `seconds` for each way of `levels`. Right: the extra
    distance boosting gives over a minute (boost speed / 60 each frame, summed)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6))
    colors = plt.get_cmap("tab10").colors
    for i, (label, pattern, style) in enumerate(levels(v)):
        trace = run(v, pattern(), 60)
        t = [k / FPS for k in range(len(trace))]
        speed = [s for _, s, _ in trace]
        width = 2.2 if i < 2 else 1.0
        left.plot(t[: seconds * FPS], speed[: seconds * FPS], style, color=colors[i], linewidth=width, label=label)
        ahead, total = [], 0.0
        for s in speed:
            total += s / FPS
            ahead.append(total)
        right.plot(t, ahead, style, color=colors[i], linewidth=width)
        right.annotate(f"{total:,.0f}", (60, total), textcoords="offset points", xytext=(4, 0), fontsize=8, color=colors[i], va="center")
    left.set(title=f"{v['name']}: boost speed over time", xlabel="time (s)", ylabel="boost speed (extra speed)")
    left.grid(alpha=0.3)
    right.set(title=f"{v['name']}: extra distance from boosting", xlabel="time (s)", ylabel="extra distance (units)")
    right.set_xlim(0, 66)
    right.grid(alpha=0.3)
    fig.legend(loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(file, dpi=100)
    plt.close(fig)
    print(f"wrote {file}")


def levels_table():
    for v in (ARWING, BLUE_MARINE, LANDMASTER):
        print(f"\n{v['name']}: each way to boost, boost held from a standstill")
        for label, pattern, _ in levels(v):
            trace = run(v, pattern(), 60)
            print(f"  {label:58}: average boost speed {average(trace):6.1f}, extra distance after 60 s {sum(s for _, s, _ in trace) / FPS:9,.0f}")


def draw(plt):
    chart(plt, ARWING, "arwing.png", 6, [("let go at 1/3 meter (30 frames)", boost_for(30)), ("hold until the meter is full", hold_always)], 90, [(30, "best: 33%"), (90, "100%")])
    chart(plt, LANDMASTER, "landmaster.png", 9, [("hold the whole time (100%)", hold_always), ("let go at half meter", boost_for(90)), ("let go at 1/3 meter", boost_for(60))], 180, [(60, "33%"), (90, "50%"), (180, "best: 100%")])
    chart(
        plt,
        BLUE_MARINE,
        "bluemarine.png",
        30,
        [("by hand: long start, then quick flicks", Player("anticipate")), ("let go at 1/3 meter", boost_for(45)), ("hold the whole time", hold_always)],
        135,
        [(3, "shortest"), (34, "25%"), (45, "33%"), (135, "100%")],
    )
    aquas_chart(plt)
    levels_chart(plt, ARWING, "arwing-levels.png", 10)
    levels_chart(plt, BLUE_MARINE, "bluemarine-levels.png", 25)
    levels_chart(plt, LANDMASTER, "landmaster-levels.png", 20)


def table(v, ns):
    print(f"\n{v['name']}: average boost speed for each boost length (then let go until boost is back)")
    full = full_meter_frames(v)
    for n in ns:
        meter = min(n, full) / full * 100
        print(f"  boost {n:3} frames ({n / FPS:4.2f} s, meter {meter:5.1f}% full): average boost speed {average(run(v, boost_for(n))):6.1f}")


def aquas_table():
    print(f"\nBlue-Marine on Aquas: seconds from the first boost to the boss arena ({AQUAS:,} units)")
    for name, pattern in [
        ("frame-perfect: let go once boost speed is 296 or more", until_speed(296)),
        ("frame-perfect: let go once boost speed is 300", until_speed(300)),
        ("frame-perfect: 1-frame boosts", boost_for(1)),
        ("frame-perfect: 39-frame boosts (29% of the meter)", boost_for(39)),
        ("hold the whole time", hold_always),
    ]:
        print(f"  {name:62}: {aquas_time(pattern):6.1f} s")
    print("By hand, average of 16 players (long start: let go at 1/2, 1/3, then 1/4 of the meter):")
    for name, mode, player in [
        ("long start, then flick 0.1 s after each boost restarts", "anticipate", {}),
        ("long start, then flick 5 times a second without looking", "rhythm", {}),
        ("long start, then flick 4 times a second without looking", "rhythm", dict(every=15)),
        ("long start, then flick when you see the boost restart", "react", {}),
        ("no long start, flick 0.1 s after each boost restarts", "anticipate", dict(start=())),
        ("no long start, flick when you see the boost restart", "react", dict(start=())),
        ("no long start, flick 2 frames every 9 (6.7 a second)", "rhythm", dict(start=(), every=9, flick=2)),
        ("first rule, flicking 4 frames too early", "anticipate", dict(aim=2)),
        ("first rule, flicking 4 frames too late", "anticipate", dict(aim=10)),
        ("first rule, timing errors of 4 frames", "anticipate", dict(sigma=4.0)),
        ("first rule, each long boost 8 frames too short", "anticipate", dict(start=(60, 37, 26))),
        ("first rule, first long boost at 2/3 of the meter", "anticipate", dict(start=(90, 45, 34))),
        ("long start, reacting in 0.28 s (17 frames)", "react", dict(reaction=17)),
        ("long start, 6-frame flicks 4 times a second without looking", "rhythm", dict(every=15, flick=6)),
        ("long start, 6-frame flicks 5 times a second without looking", "rhythm", dict(flick=6)),
    ]:
        print(f"  {name:62}: {aquas_human(mode, **player):6.1f} s")
    stacked = sum(average(run(BLUE_MARINE, Player("anticipate", seed=seed))) for seed in range(16)) / 16
    print(f"  once stacked, the first rule averages a boost speed of {stacked:.1f}")


if __name__ == "__main__":
    table(ARWING, [10, 20, 25, 29, 30, 31, 33, 35, 45, 60, 90])
    table(LANDMASTER, [10, 30, 45, 60, 90, 120, 150, 180])
    table(BLUE_MARINE, [2, 3, 6, 10, 15, 30, 34, 45, 135])
    print("\nBlue-Marine techniques:")
    for name, pattern in [
        ("hold boost the whole time", hold_always),
        ("hold, release 2 frames every 9 frames (6.7 per second)", hold_and_release(9, 2)),
        ("hold, release 3 frames every 12 frames (5 per second)", hold_and_release(12, 3)),
        ("hold, release 3 frames every 15 frames (4 per second)", hold_and_release(15, 3)),
        ("hold, release 2 frames every 6 frames (10 per second)", hold_and_release(6, 2)),
    ]:
        print(f"  {name:55}: average boost speed {average(run(BLUE_MARINE, pattern)):6.1f}")
    aquas_table()
    levels_table()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to draw the charts: pip install matplotlib)")
        raise SystemExit
    draw(plt)
