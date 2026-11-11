#!/bin/bash
# Node 17 of the HPC benchmark (#566, HANDOFF.md section 7): the kernel fixes
# of group E one by one, old against new, at threads 1, 16 and 72 (72
# replaces the handoff's 64 on Fritz).
#
#   sbatch --constraint=hwperf node17.sh 17        (submit.sh does this)
#
# Environment (submit.sh sets it): BENCH_ROOT (worktrees bench-<build>),
# BENCH_OUT (results directory), BENCH_ENV (env.sh: Julia, depot), BENCH_HPC
# (benchmarks/hpc, as sbatch runs a copy of this script).
#
# Per thread count, one process per line, each writing one TOML:
#   all       E1E2 E3 (old copies in the process) and E4 E5 E6 (new); it
#             dumps the inputs/results the variants are checked against
#   revert-E4, revert-E5, revert-E6   (bench/all with that fix reverted)
#   all       E5 E6 again (and E4 where BENCH_KF_E4_ROUNDS asks for a second
#             round), then those variants again
# so E5, E6 alternate all, variant, all, variant; E4 all, variant only by
# default: E1-E4 run on the Berlin 100 m state after 3 steps, built in every
# such process (3.3 min, measured), and a second E4 round costs about 8 min
# per thread count, which does not fit the hour. E5 and E6 run on synthetic
# 200 x 200 x 60 cases (Berlin does not run their kernels) without Berlin.
# Test runs (2026-11-10, budget 3 s): the whole sequence at 72 threads took
# 16.5 min (Berlin build + 3 steps 195 s per Berlin process); at 2 threads
# round 1 took about 16 min. Expected for 72 16 1 at budget 10 s: about
# 18 + 19 + 25 min, so about 62 min (1 thread slowest: E6 at 2-5 s a call).
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:15:00

set -u
NODE=${1:-17}
source "$BENCH_ENV"
KF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$KF/run_fixes.jl" ] || KF=$BENCH_HPC/kernel_fixes   # sbatch copies the script
O=$BENCH_OUT/node$(printf %02d "$NODE"); mkdir -p "$O"
exec > >(tee -a "$O/node.log") 2>&1
export JULIA_EXCLUSIVE=1                                    # thread i on core i
# one Aeolus cache per worktree: Julia keeps 10 per package by default, and
# the 18 builds then evict each other (seen on 2026-11-10: recompiling in jobs)
export JULIA_MAX_NUM_PRECOMPILE_FILES=${JULIA_MAX_NUM_PRECOMPILE_FILES:-100}
CFG100=$KF/../berlin_100m/berlin_100m_20250824.toml
TS=${BENCH_KF_THREADS:-"72 16 1"}
E4ROUNDS=${BENCH_KF_E4_ROUNDS:-"72:1 16:1 1:1"}             # threads:rounds of E4 (1 or 2)
BUDGET=${BENCH_KF_BUDGET:-10}                               # seconds of timing per kernel set

run() {   # run <label> <build> <threads> <fixes...> [key=value...]
    local label=$1 b=$2 t=$3; shift 3
    echo "[$(date +%T)] $label"
    julia -t "$t" --project="$BENCH_ROOT/bench-$b" "$KF/run_fixes.jl" "$O/$label.toml" "$@" \
        config="$CFG100" build="$b" label="$label" budget="$BUDGET" > "$O/$label.log" 2>&1 \
        || echo "FAILED $label"
}

{ echo "# node $NODE kernel fixes job ${SLURM_JOB_ID:-} $(date -Is)"; hostname; lscpu; numactl -H
  for b in all all-revert-E4 all-revert-E5 all-revert-E6; do git -C "$BENCH_ROOT/bench-$b" log -1 --format="$b %H %s"; done
  julia --project="$BENCH_ROOT/bench-all" -e 'using InteractiveUtils; versioninfo(verbose = false)'; } > "$O/machine.txt" 2>&1
# every build loaded once (seconds; a stale cache is compiled here, not in a run)
for b in all all-revert-E4 all-revert-E5 all-revert-E6; do
    julia --project="$BENCH_ROOT/bench-$b" -e 'using Aeolus' >> "$O/machine.txt" 2>&1 || echo "FAILED load $b"
done

for t in $TS; do
    REF=$O/kf_all_t${t}_1.toml
    N4=1; for x in $E4ROUNDS; do [ "${x%%:*}" = "$t" ] && N4=${x##*:}; done
    run kf_all_t${t}_1 all "$t" E1E2 E3 E4 E5 E6
    for e in E4 E5 E6; do run kf_revert${e}_t${t}_1 all-revert-$e "$t" $e ref="$REF"; done
    FX="E5 E6"; [ "$N4" -gt 1 ] && FX="E4 E5 E6"
    run kf_all_t${t}_2 all "$t" $FX ref="$REF"
    for e in $FX; do run kf_revert${e}_t${t}_2 all-revert-$e "$t" $e ref="$REF"; done
done
echo "[$(date +%T)] node $NODE done"
