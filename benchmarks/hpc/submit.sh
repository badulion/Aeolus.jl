#!/bin/bash
# Submit the benchmark nodes (#566, HANDOFF.md sections 6 and 8 P7).
#
#   benchmarks/hpc/submit.sh <results-dir> [node ...]     (default: every node)
#
# Nodes 1-7: node.sh (packed, several builds per node), node 17:
# kernel_fixes/node17.sh. Every job is exclusive (1 h 30 min; node 17 1 h 15 min), with
# --constraint=hwperf (LIKWID on Fritz). The worktrees bench-<build> are
# looked up under BENCH_ROOT (default: the repository's .worktrees).
set -eu
HPC=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HPC" rev-parse --show-toplevel)
OUT=$(realpath -m "$1"); shift
NODES=${*:-$(seq 1 7) 17}
export BENCH_OUT=$OUT
export BENCH_HPC=$HPC
export BENCH_ROOT=${BENCH_ROOT:-$(dirname "$(git -C "$HPC" rev-parse --git-common-dir)")/.worktrees}
export BENCH_ENV=${BENCH_ENV:-$WORK/aeolus-bench/env.sh}
export BENCH_SYSIMG=${BENCH_SYSIMG:-$WORK/aeolus-bench/sysimages}   # node.sh uses an image only if built from the worktree HEAD
mkdir -p "$OUT"
# the scripts as submitted (they are read live by the jobs: do not edit them while it runs)
mkdir -p "$OUT/scripts" && cp -r "$HPC"/*.sh "$HPC"/*.jl "$HPC"/*.py "$HPC"/kernel_fixes "$OUT/scripts/"
{ echo "submitted $(date -Is) from $(git -C "$HPC" rev-parse HEAD)"
  for d in "$BENCH_ROOT"/bench-*; do echo "$(basename "$d") $(git -C "$d" rev-parse HEAD)"; done; } > "$OUT/builds.txt"
for n in $NODES; do
    if [ "$n" = 17 ]; then s=$HPC/kernel_fixes/node17.sh; else s=$HPC/node.sh; fi
    sbatch --parsable --constraint=hwperf -J "aeolus566-n$n" -o "$OUT/slurm_node$(printf %02d "$n")_%j.out" \
        --export=ALL "$s" "$n"
done
