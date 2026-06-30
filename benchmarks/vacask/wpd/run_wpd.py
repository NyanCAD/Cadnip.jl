#!/usr/bin/env python3
# =============================================================================
# Work-Precision Diagram benchmark - VACASK side + plotting
#
# Reads the Cadnip waveforms produced by cadnip_wpd.jl, runs the real VACASK
# simulator across the same tolerance sweep through InSpice's native VACASK
# backend, computes the relative-L2 error of every run against a single golden
# reference (VACASK at tight tolerance - the more mature simulator defines
# "truth"), and emits one log-log work-precision diagram per circuit plus a
# markdown summary.
#
# Reference policy (per config.json):
#   - golden = VACASK @ ref_reltol/ref_abstol, sampled on the common grid.
#   - linear cases additionally carry an analytic waveform (analytic_<case>.csv)
#     that the golden is checked against.
#   - if VACASK is unavailable, the script degrades gracefully: golden falls back
#     to the analytic solution (linear) or the tightest Cadnip run (nonlinear),
#     and only Cadnip curves are drawn. A warning is printed.
#
# Usage:
#   pip install -r requirements.txt
#   # make the VACASK binary discoverable (see README); e.g. run ../run_vacask.sh once
#   python3 run_wpd.py [case ...]
# =============================================================================

import os
import sys
import json
import time
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VACASK_DIR = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "out")
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

with open(os.path.join(HERE, "config.json")) as f:
    CFG = json.load(f)

# Number of timed repetitions for VACASK; the minimum is reported (least noisy).
VACASK_REPEATS = int(os.environ.get("WPD_VACASK_REPEATS", "3"))


# ---------------------------------------------------------------------------
# VACASK binary discovery (mirrors run_vacask.sh's cache layout)
# ---------------------------------------------------------------------------
def locate_vacask():
    cmd = os.environ.get("VACASK_COMMAND")
    if cmd and os.path.exists(cmd):
        return cmd, os.environ.get("OSDI_PATH")
    cache = os.environ.get("CACHE_DIR", os.path.expanduser("~/.cache/cadnip-vacask"))
    cand = os.path.join(cache, "vacask", "simulator", "vacask")
    if os.path.exists(cand):
        return cand, os.environ.get("OSDI_PATH", os.path.join(cache, "vacask", "simulator"))
    return None, None


VACASK_COMMAND, OSDI_PATH = locate_vacask()


# ---------------------------------------------------------------------------
# Circuit builders (InSpice). These must match what Cadnip runs, i.e. the
# benchmarks/vacask/<case>/cedarsim/runme.sp netlists.
# ---------------------------------------------------------------------------
def build_filter():
    from InSpice import Circuit
    from InSpice.Unit import u_V, u_Ohm, u_F, u_H, u_Hz
    c = Circuit("butterworth_filter")
    # SIN(0, 1, 1/(2*pi)) -> omega = 1 rad/s
    c.SinusoidalVoltageSource("1", "vin", c.gnd, amplitude=1 @ u_V,
                              frequency=(1.0 / (2.0 * np.pi)) @ u_Hz)
    c.L("1", "vin", "n1", 1.5 @ u_H)
    c.C("2", "n1", c.gnd, (4.0 / 3.0) @ u_F)
    c.L("3", "n1", "vout", 0.5 @ u_H)
    c.R("4", "vout", c.gnd, 1.0 @ u_Ohm)
    return c


def build_graetz():
    from InSpice import Circuit
    from InSpice.Unit import u_V, u_Ohm, u_F, u_Hz
    c = Circuit("graetz_bridge")
    c.SinusoidalVoltageSource("s", "inp", "inn", amplitude=20 @ u_V, frequency=50 @ u_Hz)
    c.model("d1n4007", "D", IS=76.9e-12, RS=42.0e-3, CJO=26.5e-12, M=0.333, N=1.45)
    c.Diode("1", "inp", "outp", model="d1n4007")
    c.Diode("2", "outn", "inp", model="d1n4007")
    c.Diode("3", "inn", "outp", model="d1n4007")
    c.Diode("4", "outn", "inn", model="d1n4007")
    c.C("l", "outp", "outn", 100e-6 @ u_F)
    c.R("l", "outp", "outn", 1e3 @ u_Ohm)
    c.R("gnd1", "inn", c.gnd, 1e6 @ u_Ohm)
    c.R("gnd2", "outn", c.gnd, 1e6 @ u_Ohm)
    return c


def build_mul():
    from InSpice import Circuit
    from InSpice.Unit import u_V, u_Ohm, u_F, u_Hz
    c = Circuit("voltage_multiplier")
    # SIN(0, 50, 100k, 0, 0, 90): phase=90 deg starts at the peak (dV/dt=0).
    # phase= requires InSpice's SIN phase support; if unavailable this build
    # raises and the case is skipped with a warning.
    c.SinusoidalVoltageSource("s", "a", c.gnd, amplitude=50 @ u_V,
                              frequency=100e3 @ u_Hz, phase=90)
    c.model("diode", "D", IS=76.9e-12, RS=42.0e-3, CJO=26.5e-12, M=0.333, N=1.45)
    c.R("1", "a", "1", 0.01 @ u_Ohm)
    c.C("1", "1", "2", 100e-9 @ u_F)
    c.Diode("1", c.gnd, "1", model="diode")
    c.C("2", c.gnd, "10", 100e-9 @ u_F)
    c.Diode("2", "1", "10", model="diode")
    c.C("3", "1", "2", 100e-9 @ u_F)
    c.Diode("3", "10", "2", model="diode")
    c.C("4", "10", "20", 100e-9 @ u_F)
    c.Diode("4", "2", "20", model="diode")
    return c


BUILDERS = {"filter": build_filter, "graetz": build_graetz, "mul": build_mul}


# ---------------------------------------------------------------------------
# Running VACASK through InSpice
# ---------------------------------------------------------------------------
def vacask_simulator(circuit):
    kwargs = {"simulator": "vacask"}
    if VACASK_COMMAND:
        kwargs["vacask_command"] = VACASK_COMMAND
    if OSDI_PATH:
        kwargs["osdi_path"] = OSDI_PATH
    obj = circuit.simulator(**kwargs)
    # InSpice's circuit.simulator() may return either a Simulation (exposing
    # .transient) or a lower-level Simulator (exposing .simulation(circuit)).
    if hasattr(obj, "transient"):
        return obj
    if hasattr(obj, "simulation"):
        return obj.simulation(circuit)
    return obj


def node_signal(analysis, out_nodes):
    """Pull the output signal (single node or differential) from an InSpice analysis."""
    def get(name):
        for key in (name, name.lower(), name.upper()):
            try:
                return np.array(analysis[key])
            except (KeyError, IndexError, TypeError):
                continue
        # node-access fallback
        try:
            return np.array(analysis.nodes[name])
        except Exception as exc:  # noqa: BLE001
            raise KeyError(f"node '{name}' not in VACASK analysis") from exc

    if len(out_nodes) == 1:
        return get(out_nodes[0])
    return get(out_nodes[0]) - get(out_nodes[1])


def run_vacask(case, reltol, abstol, tspan, n_grid, out_nodes, grid):
    """Return (signal_on_grid, runtime_s) or (None, None) on failure."""
    circuit = BUILDERS[case]()
    step = (tspan[1] - tspan[0]) / n_grid
    best = None
    sig = None
    for _ in range(VACASK_REPEATS):
        sim = vacask_simulator(circuit)
        try:
            sim.options(reltol=reltol, abstol=abstol, vntol=abstol)
        except Exception:  # noqa: BLE001 - some backends take tolerances elsewhere
            pass
        t0 = time.perf_counter()
        analysis = sim.transient(step_time=step, end_time=tspan[1])
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
        t = np.array(analysis.time)
        v = node_signal(analysis, out_nodes)
        order = np.argsort(t)
        sig = np.interp(grid, t[order], v[order])
    return sig, best


# ---------------------------------------------------------------------------
# Loading Cadnip waveforms / summaries
# ---------------------------------------------------------------------------
def load_csv_wave(path, grid):
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = np.atleast_1d(data["t"])
    v = np.atleast_1d(data["v"])
    if t.size < 2:
        return None
    order = np.argsort(t)
    return np.interp(grid, t[order], v[order])


def load_cadnip_summary(case):
    """Return dict (solver, reltol_str) -> {time, steps, rejects, nniter, retcode}."""
    path = os.path.join(OUT, f"cadnip_{case}.csv")
    rows = {}
    if not os.path.exists(path):
        return rows
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    data = np.atleast_1d(data)
    for r in data:
        solver = str(r["solver"])
        reltol = float(r["reltol"])
        rows[(solver, reltol)] = {
            "time": float(r["median_time_s"]),
            "steps": int(r["steps"]),
            "retcode": str(r["retcode"]),
        }
    return rows


def reltol_tag(r):
    return "%.0e" % r  # matches Julia @sprintf("%.0e", r)


# ---------------------------------------------------------------------------
# Error metric
# ---------------------------------------------------------------------------
def rel_l2(sig, ref):
    denom = np.sqrt(np.mean(ref ** 2))
    if denom == 0 or sig is None:
        return np.nan
    return np.sqrt(np.mean((sig - ref) ** 2)) / denom


# ---------------------------------------------------------------------------
# Per-case driver
# ---------------------------------------------------------------------------
def process_case(case):
    spec = CFG["cases"][case]
    tspan = [float(x) for x in spec["tspan"]]
    n_grid = int(CFG["n_grid"])
    grid = np.linspace(tspan[0], tspan[1], n_grid)
    out_nodes = [str(x) for x in spec["output"]]
    reltols = [float(x) for x in CFG["reltols"]]
    ascale = float(CFG["abstol_scale"])
    ref_reltol = float(CFG["ref_reltol"])
    ref_abstol = float(CFG["ref_abstol"])

    print(f"\n=== {spec['title']} ({case}) ===")

    have_vacask = VACASK_COMMAND is not None and case in BUILDERS
    # try a build to catch unsupported features (e.g. SIN phase) early
    if have_vacask:
        try:
            BUILDERS[case]()
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip VACASK] cannot build {case} in InSpice: {exc}")
            have_vacask = False

    analytic = None
    apath = os.path.join(OUT, f"analytic_{case}.csv")
    if os.path.exists(apath):
        analytic = load_csv_wave(apath, grid)

    # --- golden reference -------------------------------------------------
    golden = None
    golden_src = None
    if have_vacask:
        try:
            golden, _ = run_vacask(case, ref_reltol, ref_abstol, tspan, n_grid, out_nodes, grid)
            golden_src = f"VACASK @ reltol={ref_reltol:g}"
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] VACASK golden failed: {exc}")
    if golden is None and analytic is not None:
        golden, golden_src = analytic, "analytic"
    if golden is None:
        # fall back to tightest successful Cadnip run
        for r in sorted(reltols):
            for sname, _, _ in [("IDA", 0, 0), ("FBDF", 0, 0), ("Rodas5P", 0, 0)]:
                p = os.path.join(OUT, f"cadnip_{case}_{sname}_{reltol_tag(r)}.csv")
                if os.path.exists(p):
                    golden = load_csv_wave(p, grid)
                    golden_src = f"Cadnip {sname} @ reltol={r:g}"
                    break
            if golden is not None:
                break
    if golden is None:
        print(f"  [error] no golden reference available for {case}; skipping")
        return None
    print(f"  golden reference: {golden_src}")

    if analytic is not None and golden_src and golden_src.startswith("VACASK"):
        chk = rel_l2(analytic, golden)
        print(f"  golden vs analytic cross-check: rel-L2 = {chk:.2e}")

    # --- collect curves: name -> list of (error, time) --------------------
    curves = {}
    table = []  # (sim, reltol, error, time, steps)

    # Cadnip
    summary = load_cadnip_summary(case)
    for (solver, reltol), info in sorted(summary.items()):
        p = os.path.join(OUT, f"cadnip_{case}_{solver}_{reltol_tag(reltol)}.csv")
        sig = load_csv_wave(p, grid) if os.path.exists(p) else None
        err = rel_l2(sig, golden)
        t = info["time"]
        if np.isfinite(err) and np.isfinite(t):
            curves.setdefault(f"Cadnip {solver}", []).append((err, t))
        table.append((f"Cadnip {solver}", reltol, err, t, info["steps"]))

    # VACASK sweep
    if have_vacask:
        for r in reltols:
            a = r * ascale
            try:
                sig, t = run_vacask(case, r, a, tspan, n_grid, out_nodes, grid)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] VACASK reltol={r:g} failed: {exc}")
                continue
            err = rel_l2(sig, golden)
            if np.isfinite(err) and t is not None:
                curves.setdefault("VACASK", []).append((err, t))
            table.append(("VACASK", r, err, t, np.nan))

    plot_case(case, spec["title"], curves)
    return case, golden_src, table


# ---------------------------------------------------------------------------
# Plotting + reporting
# ---------------------------------------------------------------------------
def plot_case(case, title, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in sorted(curves):
        pts = sorted(curves[name])  # by error
        if not pts:
            continue
        err = [p[0] for p in pts]
        t = [p[1] for p in pts]
        style = dict(marker="o", linewidth=1.8, markersize=5)
        if name == "VACASK":
            style.update(color="black", linestyle="--", marker="s")
        ax.plot(err, t, label=name, **style)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative L2 error vs golden")
    ax.set_ylabel("runtime [s]")
    ax.set_title(f"Work-precision: {title}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(PLOTS, f"{case}.{ext}"))
    plt.close(fig)
    print(f"  wrote {os.path.join(PLOTS, case)}.png / .svg")


def write_markdown(results):
    lines = ["# Work-Precision Diagram Results", ""]
    lines.append("Error = relative L2 of the output node vs the golden reference, "
                 "sampled on a common time grid. Each simulator runs *adaptively* "
                 "across a tolerance sweep (no forced timestep).")
    lines.append("")
    for case, golden_src, table in results:
        spec = CFG["cases"][case]
        lines.append(f"## {spec['title']}")
        lines.append("")
        lines.append(f"Golden reference: **{golden_src}**. Plot: `plots/{case}.png`.")
        lines.append("")
        lines.append("| Simulator | reltol | rel-L2 error | runtime (s) | steps |")
        lines.append("|-----------|--------|--------------|-------------|-------|")
        for sim, reltol, err, t, steps in table:
            es = "-" if not np.isfinite(err) else f"{err:.2e}"
            ts = "-" if t is None or not np.isfinite(t) else f"{t:.3f}"
            ss = "-" if steps is None or (isinstance(steps, float) and np.isnan(steps)) else str(int(steps))
            lines.append(f"| {sim} | {reltol:g} | {es} | {ts} | {ss} |")
        lines.append("")
    md = "\n".join(lines)
    path = os.path.join(OUT, "wpd_results.md")
    with open(path, "w") as f:
        f.write(md)
    print(f"\nMarkdown summary: {path}")


def main():
    if VACASK_COMMAND is None:
        print("[warn] VACASK binary not found. Set VACASK_COMMAND (and OSDI_PATH), or run "
              "../run_vacask.sh once to populate the cache. Proceeding with Cadnip-only "
              "curves and analytic/tightest-Cadnip golden where possible.")
    else:
        print(f"VACASK: {VACASK_COMMAND}")
        if OSDI_PATH:
            print(f"OSDI path: {OSDI_PATH}")

    cases = sys.argv[1:] if len(sys.argv) > 1 else list(CFG["cases"].keys())
    results = []
    for case in cases:
        if case not in CFG["cases"]:
            print(f"[warn] unknown case {case}")
            continue
        res = process_case(case)
        if res is not None:
            results.append(res)
    if results:
        write_markdown(results)


if __name__ == "__main__":
    main()
