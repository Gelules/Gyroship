"""Star Fox (2026) boost math: a frame-by-frame simulation of the boost meter
and the boost speed of the Arwing, the Landmaster and the Blue-Marine, at 60 fps.

The values were datamined from the game files (each vehicle's boost script and
the player vehicle's own settings) and checked frame by frame against the game
running its own code.

Run:  python boost_math.py
It prints the tables of the write-up. With matplotlib installed
(pip install matplotlib) it also draws the charts as PNG files.
"""

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


class Landmaster:
    """The Landmaster's boost speed eases toward 300 (fast at first, capped
    at 3 per frame, then slower and slower) and fades by 5% per frame when not
    boosting. The meter fills and drains at the same rate; when you let go
    before the meter is full, the lock starts one frame before it drains."""

    def __init__(self, v):
        self.v = v
        self.rate = f32(v["rate"] * DT)
        self.k = f32(30 * DT)  # 0.5 at 60 fps
        self.meter, self.speed, self.locked = 0.0, 0.0, False

    def ease(self, target, share, cap):
        step = f32(share * (target - self.speed))
        if abs(step) >= 0.001:
            step = max(-cap, min(cap, step))
        else:
            step = 0.001 if target != self.speed else 0.0
        self.speed = f32(self.speed + f32(self.k * step))

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


def draw(plt):
    chart(plt, ARWING, "arwing.png", 6, [("let go at 1/3 meter (30 frames)", boost_for(30)), ("hold until the meter is full", hold_always)], 90, [(30, "best: 33%"), (90, "100%")])
    chart(plt, LANDMASTER, "landmaster.png", 9, [("hold the whole time (100%)", hold_always), ("let go at half meter", boost_for(90)), ("let go at 1/3 meter", boost_for(60))], 180, [(60, "33%"), (90, "50%"), (180, "best: 100%")])
    chart(
        plt,
        BLUE_MARINE,
        "bluemarine.png",
        30,
        [("hold, quick release ~6-7 times a second", hold_and_release(9, 2)), ("let go at 1/3 meter", boost_for(45)), ("hold the whole time", hold_always)],
        135,
        [(3, "shortest"), (34, "25%"), (45, "33%"), (135, "100%")],
    )


def table(v, ns):
    print(f"\n{v['name']}: average boost speed for each boost length (then let go until boost is back)")
    full = full_meter_frames(v)
    for n in ns:
        meter = min(n, full) / full * 100
        print(f"  boost {n:3} frames ({n / FPS:4.2f} s, meter {meter:5.1f}% full): average boost speed {average(run(v, boost_for(n))):6.1f}")


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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to draw the charts: pip install matplotlib)")
        raise SystemExit
    draw(plt)
