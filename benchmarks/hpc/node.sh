#!/bin/bash
# One node of the HPC benchmark (#566, HANDOFF.md section 6), as adapted on
# Fritz: 72 cores (2 × 36, 4 NUMA domains, no SMT), so 72 threads where the
# handoff says 64 and 36 where it says 32; group F (#563) dropped and its
# slot used by bench/only-543 (#545 + #543, decided 2026-11-10).
#
#   sbatch --constraint=hwperf node.sh <node>      (submit.sh does this; hwperf: LIKWID)
#
# Environment (submit.sh sets it): BENCH_ROOT (worktrees bench-<build>),
# BENCH_OUT (results directory), BENCH_ENV (env.sh: Julia, depot),
# BENCH_SYSIMG (system images, sysimage/build.jl).
#
# Every node: the machine record, then the reference block (STREAM at the
# node's thread counts; 10 steps of bench/master on Berlin 100 m at 72
# threads), then its matrix. Each run is one process writing one TOML.
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:30:00

set -u
NODE=$1
source "$BENCH_ENV"
HPC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$HPC/bench_run.jl" ] || HPC=$BENCH_HPC        # sbatch copies the script; BENCH_HPC points at the originals
CFG100=$HPC/berlin_100m/berlin_100m_20250824.toml
CFG50=$HPC/berlin_50m/berlin_50m_20250824.toml
BUDGET=${BENCH_BUDGET:-300}                          # seconds of measured stepping per process

# ---- the matrix -------------------------------------------------------------
# Packed (decided 2026-11-10): with the system images a build at 16 and 72
# threads takes about 20 min, so several builds share a node. They run one
# after the other, never concurrently (they would share memory bandwidth),
# and share the node's reference. Entry: build|threads|extras at 72 threads.
# Nodes 1 and 2 (1-8 threads) leave most cores idle by design: the full
# scaling curve was kept (maintainer, 2026-11-10). Node 17 is
# kernel_fixes/node17.sh.
case $NODE in
  1) M=("master|1 2 4 8|") ;;
  2) M=("all|1 2 4 8|") ;;
  3) M=("master|16 36 72|trace counters" "all|16 36 72|trace counters numa") ;;
  4) M=("only-A|16 72|trace counters" "only-B|16 72|trace counters" "only-C|16 72|trace counters") ;;
  5) M=("only-D|16 72|trace counters" "only-E|16 72|trace counters" "only-543|16 72|trace counters") ;;
  6) M=("all-minus-A|16 72|trace counters" "all-minus-B|16 72|trace counters" "all-minus-C|16 72|trace counters") ;;
  7) M=("all-minus-D|16 72|trace counters" "all-minus-E|16 72|trace counters") ;;
  dry*) M=("$BENCH_DRY_BUILD|2 72|trace counters") ;;    # the dry run (P6): every metric path once
  *)  echo "node $NODE: not in the matrix (1-7; 17 is kernel_fixes/node17.sh)"; exit 2 ;;
esac
ALL_TS=$(for e in "${M[@]}"; do e=${e#*|}; echo "${e%%|*} 72"; done | tr ' ' '\n' | sort -nu | tr '\n' ' ')
[ "${BENCH_MATRIX_ONLY:-0}" = 1 ] && { printf '%s\n' "${M[@]}"; echo "stream at: $ALL_TS"; exit 0; }

case $NODE in [0-9]*) O=$BENCH_OUT/node$(printf %02d "$NODE") ;; *) O=$BENCH_OUT/node_$NODE ;; esac; mkdir -p "$O"
exec > >(tee -a "$O/node.log") 2>&1
export JULIA_EXCLUSIVE=1                             # thread i on core i

# the build's system image (sysimage/build.jl) if BENCH_SYSIMG holds one built
# from the worktree's current commit; else none, and the run compiles as usual
sysimg() {   # sysimg <build>  -> prints "-J <file>" or nothing
    local f=${BENCH_SYSIMG:-}/$1.so
    [ -n "${BENCH_SYSIMG:-}" ] && [ -f "$f" ] && [ "$(cat "$f.commit" 2>/dev/null)" = "$(git -C "$BENCH_ROOT/bench-$1" rev-parse HEAD)" ] \
        && echo "-J $f"
}

run() {   # run <build> <label> <threads> <mode> <cfg> <dt> <case> [prefix...] -- [keys...]
    local b=$1 label=$2 t=$3 mode=$4 cfg=$5 dt=$6 cs=$7; shift 7
    local pre=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do pre+=("$1"); shift; done; shift
    local J; J=$(sysimg "$b")
    echo "[$(date +%T)] $label ${J:-(no system image)}"
    local args=(julia $J -t "$t" --project="$BENCH_ROOT/bench-$b" "$HPC/bench_run.jl" "$mode" "$cfg" "$dt" "$O/$label.toml"
                build="$b" case="$cs" label="$label" budget="$BUDGET" "$@")
    "${pre[@]}" "${args[@]}" && return
    echo "FAILED $label"
    # a failure under likwid-perfctr is retried once without it: the timings
    # matter more than the counters (the timeline file is kept for diagnosis)
    if [ ${#pre[@]} -gt 0 ] && [[ "${pre[0]}" == *likwid_wrap.sh ]]; then
        echo "[$(date +%T)] $label again without likwid"
        "${args[@]}" || echo "FAILED $label (without likwid)"
    fi
}
cfg_of() { [ "$1" = 100m ] && echo "$CFG100 2.0" || echo "$CFG50 1.0"; }   # cfg_of <case> -> "<config> <dt>"

# ---- the machine ----------------------------------------------------------------
B0=${M[0]%%|*}
{ echo "# node $NODE job ${SLURM_JOB_ID:-} $(date -Is)"; printf '#   %s\n' "${M[@]}"; hostname; lscpu; numactl -H
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
  echo "no_turbo $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"
  for e in "${M[@]}"; do b=${e%%|*}; echo "$b $(git -C "$BENCH_ROOT/bench-$b" log -1 --format='%H %s')"; done
  module load likwid >/dev/null 2>&1 && likwid-topology -g 2>/dev/null | head -60
  julia --project="$BENCH_ROOT/bench-$B0" -e 'using InteractiveUtils; versioninfo(verbose = false)'; } > "$O/machine.txt" 2>&1

# ---- the reference block --------------------------------------------------------
for t in $ALL_TS; do
    julia $(sysimg "$B0") -t "$t" --project="$BENCH_ROOT/bench-$B0" "$HPC/roofline_cfg.jl" stream > "$O/stream_t$t.txt" 2>&1 || echo "FAILED stream t$t"
done
run master ref_master_100m_t72 72 time "$CFG100" 2.0 100m -- rounds=1 nsteps=10

# ---- the matrix -----------------------------------------------------------------
# One process per (build, case, threads), label time_<case>_t<threads>_<build>.
# At 72 threads the same process also runs the launch trace after its timed
# blocks (trace=1), and with counters it runs under likwid-perfctr's timeline
# (cut to the timed blocks by summary.py).
HAVE_LIKWID=0
if [[ " ${M[*]} " == *counters* ]]; then
    if module load likwid >/dev/null 2>&1 && command -v likwid-perfctr >/dev/null; then
        HAVE_LIKWID=1
        # Ice Lake SP has no NUMA group: MEM (per socket) and UPI (cross-socket) show placement
        export LIKWID_GROUPS="-g MEM -g FLOPS_DP -g L3 -g UPI -g TMA -g ENERGY"
    else
        echo "counters: likwid-perfctr not available (module or --constraint=hwperf missing)" > "$O/counters_skipped.txt"
    fi
fi
for e in "${M[@]}"; do
    b=${e%%|*}; e=${e#*|}; TS=${e%%|*}; X=${e#*|}
    for t in $TS; do
        for cs in ${BENCH_CASES:-100m 50m}; do
            read -r cfg dt <<< "$(cfg_of $cs)"
            lab=time_${cs}_t${t}_$b; pre=(); keys=()
            if [ "$t" = 72 ] && [[ " $X " == *" trace "* ]]; then keys+=(trace=1); fi
            if [ "$t" = 72 ] && [[ " $X " == *" counters "* ]] && [ $HAVE_LIKWID = 1 ]; then
                date +%s.%N > "$O/$lab.t0"
                pre=("$HPC/likwid_wrap.sh" "$O/$lab.timeline")
            fi
            run "$b" "$lab" "$t" time "$cfg" "$dt" "$cs" "${pre[@]}" -- "${keys[@]}"
        done
    done
    if [[ " $X " == *" numa "* ]]; then          # first touch from the main thread against interleaved pages
        for cs in ${BENCH_CASES:-100m 50m}; do
            read -r cfg dt <<< "$(cfg_of $cs)"
            BENCH_NUMACTL="--interleave=all" run "$b" numa_interleave_${cs}_t72_$b 72 time "$cfg" "$dt" "$cs" \
                numactl --interleave=all --
        done
    fi
done
echo "[$(date +%T)] node $NODE done"
