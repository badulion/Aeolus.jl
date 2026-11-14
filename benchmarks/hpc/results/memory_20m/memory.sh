#!/bin/bash
# Why RSS grows during stepping (#566, 20 m rerun): bench_run.jl memory mode
# on 20 m at 72 threads, then 50 m at 72 and 8 threads (what scales with N).
# Unpinned (allocation does not depend on pinning; runs faster).
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -o /home/atuin/b337bd/b337bd10/aeolus-bench/scaling20/memory_%j.out
W=/home/hpc/b337bd/b337bd10/smarttwin/Aeolus.jl/.worktrees
A=/home/atuin/b337bd/b337bd10/aeolus-bench
H=$W/issue-566/benchmarks/hpc
source $A/env.sh
unset JULIA_EXCLUSIVE
OUT=$A/results/memory_20m; mkdir -p $OUT
echo "[$(date -Is)] $(hostname), bench-all $(git -C $W/bench-all rev-parse --short HEAD), scripts $(git -C $H rev-parse --short HEAD)"
julia -e 'Meta.parseall(read(ARGS[1], String)); println("bench_run.jl parses")' $H/bench_run.jl || exit 1
for c in "20m berlin_20m/berlin_20m_20250824.toml 0.4 72" "50m berlin_50m/berlin_50m_20250824.toml 1.0 72" "50m berlin_50m/berlin_50m_20250824.toml 1.0 8"; do
  set -- $c
  echo "[$(date +%T)] memory $1 t$4"
  julia -t $4,0 --project=$W/bench-all $H/bench_run.jl memory $H/$2 $3 $OUT/memory_$1_t$4_all.toml \
      build=all case=$1 label=memory_$1_t$4_all nwarm=2 nmem=4 sample_rate=0.01 budget=900 || echo "FAILED $1 t$4"
  cat $OUT/memory_$1_t$4_all.memory.txt 2>/dev/null | head -30
done
echo "[$(date -Is)] done"
