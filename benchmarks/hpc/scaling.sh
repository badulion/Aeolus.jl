#!/bin/bash
# The scaling test of #566 (after run 1): Berlin 20 m (berlin_20m/, 150 M grid
# points) with bench/all, one process per thread count, on one node.
#
#   sbatch --constraint=hwperf scaling.sh <results-dir> [threads ...]
#        default threads: 72 36 16 8 4 2 1 (largest first: if the job runs out
#        of time, the slow low-thread runs are the ones lost)
#
# Changes against run 1 (the analysis of run 1, 2026-11-11):
#   * -t N,0: no interactive thread, so the main task runs on thread 1, which
#     JULIA_EXCLUSIVE=1 pins to core 0. With -t N (N > 1) Julia 1.12+ adds an
#     interactive thread and the main task, i.e. all serial code, ran on an
#     unpinned OS thread (affinity 0-71); the likely cause of 2 threads being
#     slower than 1 in run 1.
#   * FFTW on Threads.nthreads() threads: the 20 m config has no [fftw]
#     threads key (run 1's configs fixed it at the laptop's 12).
#   * no launch trace: in run 1 the trace recorded the ~20 000 launches
#     nested inside the halo fills per step and added about as much time as
#     the halo fills it reported (0.21 of 0.47 s at 100 m, 0.61 of 1.46 s at
#     50 m). Timed steps only.
#   * /usr/bin/time -v around each process: peak memory (maximum resident set
#     size). A 100 m process on the laptop used 4.0 GiB peak and 1.84 GiB live
#     heap after 3 steps, about 0.7 kB live and 1.4 kB peak per storage point;
#     150 M points so need about 110 GB live and about 200 GB peak of the
#     node's 256 GB. If the 72-thread process runs out of memory, stop.
#   * summary: mean and median per step and GC time (run 1 reported medians
#     only, which hid the GC pauses at 50 m).
#
# Environment: BENCH_ROOT (worktrees bench-<build>), BENCH_ENV (env.sh),
# BENCH_BUILD (default all), BENCH_BUDGET (seconds of measured stepping per
# process, default 900).
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 24:00:00

set -u
OUT=$(realpath -m "$1"); shift
TS=${*:-72 36 16 8 4 2 1}
source "$BENCH_ENV"
HPC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$HPC/bench_run.jl" ] || HPC=$BENCH_HPC
B=${BENCH_BUILD:-all}
CFG=$HPC/berlin_20m/berlin_20m_20250824.toml
BUDGET=${BENCH_BUDGET:-900}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/scaling.log") 2>&1
export JULIA_EXCLUSIVE=1
# GNU time where it exists; Fritz's compute nodes have none (timev.py: the same report)
if [ -x /usr/bin/time ]; then TIMEV=(/usr/bin/time -v); else
    command -v python3 >/dev/null || module load python/3.13-miniforge >/dev/null 2>&1
    TIMEV=(python3 "$HPC/timev.py"); fi
{ echo "# scaling job ${SLURM_JOB_ID:-} $(date -Is) threads: $TS"; hostname; lscpu; numactl -H; free -g
  echo "bench-$B $(git -C "$BENCH_ROOT/bench-$B" log -1 --format='%H %s')"
  echo "scripts $(git -C "$HPC" rev-parse HEAD)"; } > "$OUT/machine.txt" 2>&1

for t in $TS; do
    lab=scaling_20m_t${t}_$B
    echo "[$(date +%T)] $lab"
    "${TIMEV[@]}" -o "$OUT/$lab.time.txt" \
        julia -t "$t,0" --project="$BENCH_ROOT/bench-$B" "$HPC/bench_run.jl" time "$CFG" 0.4 "$OUT/$lab.toml" \
            build="$B" case=20m label="$lab" budget="$BUDGET" nwarm=2 rounds=1 nsteps=5 \
        || echo "FAILED $lab (exit $?)"
    grep -E "Maximum resident|Elapsed" "$OUT/$lab.time.txt" 2>/dev/null
done
echo "[$(date +%T)] scaling done"

# the summary: per thread count the build time, peak memory, median and mean
# s/step per setting, GC share, and speed-up and efficiency against 1 thread
julia --project="$BENCH_ROOT/bench-$B" -e '
using TOML, Statistics
out = ARGS[1]
rows = []
for f in readdir(out; join = true)
    endswith(f, ".toml") && occursin("scaling_20m_t", f) || continue
    d = TOML.parsefile(f)
    rss = let tf = replace(f, ".toml" => ".time.txt"), m = isfile(tf) ? match(r"Maximum resident set size \(kbytes\): (\d+)", read(tf, String)) : nothing
        m === nothing ? NaN : parse(Int, m[1]) / 2^20 end
    push!(rows, (t = d["threads"], build = d["t_build_s"], rss = rss,
                 ip = get(d, "steps_if_possible", Float64[]), nv = get(d, "steps_never", Float64[]),
                 gc = d["gc_s"], wall = d["measured_wall_s"]))
end
sort!(rows; by = r -> r.t)
isempty(rows) && exit()
base = first(rows).t == 1 ? median(first(rows).ip) : NaN
open(joinpath(out, "summary.md"), "w") do io
    println(io, "| threads | build s | peak GB | if_possible median | mean | never median | GC % | speed-up vs 1 | efficiency |")
    println(io, "| --: | --: | --: | --: | --: | --: | --: | --: | --: |")
    for r in rows
        m = median(r.ip); s = base / m
        println(io, "| $(r.t) | $(round(Int, r.build)) | $(round(r.rss; digits = 1)) | $(round(m; digits = 3)) | ",
                "$(round(mean(r.ip); digits = 3)) | $(isempty(r.nv) ? "-" : round(median(r.nv); digits = 3)) | ",
                "$(round(100r.gc / r.wall; digits = 1)) | $(round(s; digits = 2)) | $(round(100s / r.t; digits = 0)) % |")
    end
end
print(read(joinpath(out, "summary.md"), String))' "$OUT"
