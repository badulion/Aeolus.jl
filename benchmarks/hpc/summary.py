#!/usr/bin/env python3
"""Summary of the HPC benchmark (#566, HANDOFF.md section 9): tables and figures.

    python3 summary.py <results-dir>        (Fritz: module load python/3.13-miniforge)

Reads every node*/ *.toml written by bench_run.jl (and node17/ *.toml, the
kernel fixes), the STREAM lines (stream_t*.txt) and the LIKWID timelines
(counters_*.timeline), and writes <results-dir>/summary.md and figures
<results-dir>/fig_*.png. Times are reported raw and normalised to each node's
reference (20 steps of bench/master, Berlin 100 m, 72 threads): normalised =
raw × (mean reference over nodes) / (this node's reference).

The time of a build is its loop_fusion = "if_possible" median per step
("none" on master, which has no setting); "never" is reported beside it.
"""
import math, re, sys, tomllib
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(sys.argv[1])
runs, refs, stream, missing = [], {}, defaultdict(dict), []
for f in sorted(R.glob("node*/*.toml")):
    try:
        d = tomllib.loads(f.read_text())
    except Exception as e:                       # a TOML that does not parse is reported, not skipped silently
        missing.append(f"{f.relative_to(R)}: does not parse ({e})"); continue
    d["_node"] = f.parent.name; d["_file"] = f.name
    if f.parent.name == "node17":
        d["_kernel_fix"] = True
    runs.append(d)
    if d.get("label", "").startswith("ref_master"):
        refs[d["_node"]] = d.get("step_median_none")
for f in sorted(R.glob("node*/stream_t*.txt")):
    for l in f.read_text().splitlines():
        m = re.match(r"ROW \S+ (\d+) stream_(\w+)_(\w+) ([\d.eE+-]+)", l)
        if m:
            stream[(f.parent.name, int(m[1]))][f"{m[2]}_{m[3]}"] = float(m[4])

ref_mean = sum(v for v in refs.values() if v) / max(1, sum(1 for v in refs.values() if v))
def norm(node, t):
    r = refs.get(node)
    return t * ref_mean / r if (r and t is not None) else None

def best(d):        # the build's time per step: if_possible, else none (master)
    for k in ("if_possible", "none"):
        if f"step_median_{k}" in d:
            return d[f"step_median_{k}"], k
    return None, None

T = {}              # (build, case, threads) -> dict
for d in runs:
    if d.get("mode") != "time" or d.get("_kernel_fix") or not d["label"].startswith("time_"):
        continue
    t, k = best(d)
    T[(d["build"], d["case"], d["threads"])] = dict(
        raw=t, norm=norm(d["_node"], t), setting=k, node=d["_node"],
        never=d.get("step_median_never"), mn=d.get(f"step_min_{k}"),
        gc=d.get("gc_s", 0.0) / max(1, sum(len(d.get(f"steps_{s}", [])) for s in ("never", "if_possible", "none"))),
        nsteps=d.get("nsteps"), rounds=d.get("rounds"))

out = ["# #566 HPC benchmark: summary", "",
       f"Results: `{R}`. Reference (bench/master, 100 m, 72 threads, median s/step) per node: " +
       ", ".join(f"{n} {v:.3f}" for n, v in sorted(refs.items()) if v) + f"; mean {ref_mean:.3f}.", ""]
if missing:
    out += ["**Not read:**", ""] + [f"- {m}" for m in missing] + [""]

# 1. time per step
out += ["## Time per step (s, median; normalised to the node reference)", "",
        "| build | case | threads | if_possible / none | never | min | normalised | GC/step | node |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: | :-- |"]
for (b, c, t), v in sorted(T.items()):
    f = lambda x: "–" if x is None else f"{x:.3f}"
    out.append(f"| {b} | {c} | {t} | {f(v['raw'])} | {f(v['never'])} | {f(v['mn'])} | {f(v['norm'])} | {f(v['gc'])} | {v['node']} |")
out.append("")

# 2. speed-ups at 16 and 72 threads (normalised times)
def sp(a, b, c, t):     # speed-up of b over a = time(a) / time(b)
    x, y = T.get((a, c, t)), T.get((b, c, t))
    return x["norm"] / y["norm"] if x and y and x["norm"] and y["norm"] else None
groups = {"A": "only-A", "B": "only-B", "C": "only-C", "D": "only-D", "E": "only-E", "#543": "only-543"}
out += ["## Speed-ups (normalised times)", "",
        "Alone: add-one build against master, except C (only-C against only-543: #550 contains #543).",
        "Within everything: all against all-minus. Interaction: product of the alone speed-ups against all/master.", "",
        "| group | case | threads | alone | within all |", "| :-- | :-- | --: | --: | --: |"]
for c in ("100m", "50m"):
    for t in (16, 72):
        prod = 1.0
        for g, b in groups.items():
            base = "only-543" if g == "C" else "master"
            a = sp(base, b, c, t)
            w = sp(f"all-minus-{g}", "all", c, t) if g in "ABCDE" else None
            if a and g != "#543":
                prod *= a
            fm = lambda x: "–" if x is None else f"{x:.2f}×"
            out.append(f"| {g} | {c} | {t} | {fm(a)} | {fm(w)} |")
        al = sp("master", "all", c, t)
        out.append(f"| **product of A, B, C×#543… vs all** | {c} | {t} | {prod * (sp('master', 'only-543', c, t) or 1):.2f}× | all/master {al:.2f}× |" if al else f"| product | {c} | {t} | {prod:.2f}× | – |")
out += ["", "(The product uses C against only-543 times #543 against master, so #543 counts once.)", ""]

# 3. scaling figure
fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, c in zip(axs, ("100m", "50m")):
    for b, st in (("master", "o-"), ("all", "s-")):
        pts = sorted((t, v["norm"]) for (bb, cc, t), v in T.items() if bb == b and cc == c and v["norm"])
        if pts:
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], st, label=b)
            t1 = pts[0]
            ax.loglog([p[0] for p in pts], [t1[1] * t1[0] / p[0] for p in pts], ":", color=ax.lines[-1].get_color(), alpha=0.5)
    ax.set_title(f"Berlin {c}"); ax.set_xlabel("threads"); ax.set_ylabel("s per step (normalised)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
fig.savefig(R / "fig_scaling.png", dpi=120); plt.close(fig)
out += ["![scaling](fig_scaling.png)", "", "Dotted: ideal scaling from the smallest thread count.", ""]

# 4. stacked bars at 72 threads from the trace runs: kernels, idle in kernels, serial between, GC
bars = []
for d in runs:
    tr = d.get("trace_if_possible") or d.get("trace_none")      # a trace mode run, or a time run with trace=1
    if not tr:
        continue
    n = len(tr["steps"]); nt = d["threads"]
    busy = sum(tr["busy_per_thread_s"]) / nt / n
    idle = tr["idle_in_launches_s"] / nt / n
    ser = tr["serial_between_s"] / n
    gc = (T.get((d["build"], d["case"], nt)) or {}).get("gc", 0.0)
    bars.append((d["case"], d["build"], busy, idle, max(ser - gc, 0.0), gc, tr["step_wall_s"] / n))
for c in ("100m", "50m"):
    bb = sorted(b for b in bars if b[0] == c)
    if not bb:
        continue
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(bb)), 4), constrained_layout=True)
    x = range(len(bb)); bottom = [0.0] * len(bb)
    for i, (nm, col) in enumerate((("kernels (busy)", "C0"), ("idle inside launches", "C1"),
                                   ("serial between launches", "C2"), ("GC", "C3"))):
        h = [b[2 + i] for b in bb]
        ax.bar(x, h, bottom=bottom, label=nm, color=col); bottom = [p + q for p, q in zip(bottom, h)]
    ax.set_xticks(list(x)); ax.set_xticklabels([b[1] for b in bb], rotation=45, ha="right")
    ax.set_ylabel("s per step (raw, traced)"); ax.set_title(f"Berlin {c}, 72 threads, if_possible")
    ax.legend(fontsize=8)
    fig.savefig(R / f"fig_breakdown_{c}.png", dpi=120); plt.close(fig)
    out += [f"![breakdown {c}](fig_breakdown_{c}.png)", ""]
out += ["Per step, from the launch trace (one process per build with the trace on, so slower than the timed runs): "
        "kernels = busy summed over threads / threads; idle = idle inside launches / threads; serial = step wall − "
        "summed launch walls, less GC (taken from the timed run).", ""]

# 5. LIKWID counters at 72 threads, over the timed steps, against STREAM triad.
# likwid-perfctr -t writes no header into the timeline (stderr): each row is
#   <group> <nmetrics> <ncpu> <seconds since start> then nmetrics × ncpu values
# (metric-major). The groups are numbered in the order node.sh passes them
# (LIKWID_GROUPS) and the metric positions are those of LIKWID 5.5's ICX
# group files. Multiplexed every 2 s, each group is sampled every 12 s, so
# the window is the whole timed section (both settings), not one setting.
LW_GROUPS = ["MEM", "FLOPS_DP", "L3", "UPI", "TMA", "ENERGY"]
LW_METRICS = {   # name: (group, metric index, reduce over cpus, scale)
    "mem_GBs": ("MEM", 8, "sum", 1e-3), "flops_GFs": ("FLOPS_DP", 4, "sum", 1e-3),
    "upi_GBs": ("UPI", 8, "sum", 1e-3), "backend_pct": ("TMA", 9, "mean", 1.0),
    "power_W": ("ENERGY", 7, "sum", 1.0), "dram_power_W": ("ENERGY", 11, "sum", 1.0)}
def timeline(f, t0, ts0, ts1):
    acc = defaultdict(list)
    for l in f.read_text(errors="replace").splitlines():
        p = l.split()
        if len(p) < 5 or not (p[0].isdigit() and p[1].isdigit() and p[2].isdigit()):
            continue
        gid, nm, ncpu = int(p[0]), int(p[1]), int(p[2])
        if gid > len(LW_GROUPS) or len(p) != 4 + nm * ncpu or not (ts0 <= t0 + float(p[3]) <= ts1):
            continue
        g = LW_GROUPS[gid - 1]
        for k, (gg, i, red, sc) in LW_METRICS.items():
            if gg != g:
                continue
            vs = [float(v) for v in p[4 + i * ncpu: 4 + (i + 1) * ncpu]]
            acc[k].append(sc * (sum(vs) if red == "sum" else sum(vs) / len(vs)))
    return {k: (sum(v) / len(v), len(v)) for k, v in acc.items()}
out += ["## Hardware counters at 72 threads (LIKWID, over the timed steps)", "",
        "Means over the timeline samples inside the timed section (both settings; n = MEM samples). "
        "Memory and cross-socket (UPI) traffic in GB/s, double-precision GFLOP/s, back-end bound share of "
        "pipeline slots (TMA), package and DRAM power summed over both sockets.", "",
        "| build | case | memory GB/s | STREAM triad | share | UPI GB/s | GFLOP/s | back end | power W | DRAM W | n |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |"]
COUNTERS = {}
for d in sorted(runs, key=lambda d: (d.get("case", ""), d.get("build", ""))):
    node = R / d["_node"]; lab = d.get("label", "")
    tl, t0f = node / f"{lab}.timeline", node / f"{lab}.t0"
    if not tl.exists() or not t0f.exists() or "ts_begin" not in d:
        continue
    m = timeline(tl, float(t0f.read_text()), d["ts_begin"], d["ts_end"])
    COUNTERS[(d["build"], d["case"])] = m
    st = stream.get((d["_node"], 72), {}).get("triad_dram")
    g = lambda k, fmt="{:.1f}": fmt.format(m[k][0]) if k in m else "–"
    share = f"{100 * m['mem_GBs'][0] / st:.0f} %" if ("mem_GBs" in m and st) else "–"
    out.append(f"| {d['build']} | {d['case']} | {g('mem_GBs')} | {st or float('nan'):.0f} | {share} | {g('upi_GBs')} | "
               f"{g('flops_GFs')} | {g('backend_pct', '{:.0f} %')} | {g('power_W', '{:.0f}')} | {g('dram_power_W', '{:.0f}')} | "
               f"{m.get('mem_GBs', (0, 0))[1]} |")
out.append("")

# 6. NUMA: interleaved pages against first touch
out += ["## NUMA (bench/all, 72 threads)", "", "| case | first touch | --interleave=all |", "| :-- | --: | --: |"]
for d in runs:
    if d.get("label", "").startswith("numa_interleave_"):
        t, _ = best(d); base = T.get(("all", d["case"], 72), {}).get("raw")
        out.append(f"| {d['case']} | {f(base)} | {f(t)} |")
out.append("")

# 7. kernel fixes (node 17): median ms per call, old against new. E1 (update)
# and E2 (strain) and E3 have both versions in one process (bench/all);
# E4-E6 compare bench/all against bench/all with that fix reverted, the
# medians of the rounds averaged. `agree`: the variant's result matched
# within the stated tolerance (checked in the later process of each pair).
kf = [d for d in runs if d.get("_kernel_fix")]
if kf:
    KF = [("E1", "E1E2", "update"), ("E2", "E1E2", "strain"), ("E3", "E3", "pass"), ("E3", "E3", "advect"),
          ("E4", "E4", "pass"), ("E4", "E4", "n2_loop"), ("E5", "E5", "pass"), ("E6", "E6", "pass"), ("E6", "E6", "update")]
    old, new, agree, alloc = defaultdict(list), defaultdict(list), defaultdict(list), {}
    for d in kf:
        th, rev = d.get("threads"), d.get("build", "").startswith("all-revert-")
        for fx in d.get("fixes", []):
            e = d.get(fx, {}); tm = e.get("timings", {})
            if e.get("agree") is not None:
                agree[(fx, th)].append(bool(e["agree"]))
            for name, sec, key in KF:
                if sec != fx:
                    continue
                if f"{key}_old" in tm:
                    old[(name, key, th)].append(tm[f"{key}_old"]["median_ms"]); new[(name, key, th)].append(tm[f"{key}_new"]["median_ms"])
                elif key in tm:
                    (old if rev else new)[(name, key, th)].append(tm[key]["median_ms"])
            if "alloc_update_bytes" in e:
                alloc[("old" if rev else "new", th)] = e["alloc_update_bytes"]
    out += ["## Kernel fixes (node 17)", "",
            "Median ms per call; E1-E3 old and new in one process, E4-E6 bench/all against bench/all with the fix reverted.", "",
            "| fix | kernel | threads | old | new | speed-up | agree |", "| :-- | :-- | --: | --: | --: | --: | :-- |"]
    mean = lambda v: sum(v) / len(v)
    for name, sec, key in KF:
        for th in sorted({k[2] for k in new if k[:2] == (name, key)}):
            o, n = old.get((name, key, th)), new.get((name, key, th))
            if not o or not n:
                continue
            a = agree.get((sec, th), [])
            out.append(f"| {name} | {key} | {th} | {mean(o):.1f} | {mean(n):.1f} | {mean(o) / mean(n):.2f}× | "
                       f"{'yes' if a and all(a) else ('NO' if a else '–')} |")
    if alloc:
        out += ["", "E6 allocation per update call (bytes): " +
                ", ".join(f"{th} threads {alloc.get(('old', th), '–')} → {alloc.get(('new', th), '–')}" for th in sorted({k[1] for k in alloc}))]
    out.append("")

# what is missing
exp_builds = ["master", "all", "only-A", "only-B", "only-C", "only-D", "only-E", "only-543",
              "all-minus-A", "all-minus-B", "all-minus-C", "all-minus-D", "all-minus-E"]
have = {k[0] for k in T}
out += ["## Not measured", "", "- group F (#563) and its builds: dropped (maintainer, 2026-11-10)."]
out += [f"- {b}: no timed run found" for b in exp_builds if b not in have]
(R / "summary.md").write_text("\n".join(out) + "\n")
print(f"wrote {R / 'summary.md'}")
