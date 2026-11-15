#!/bin/bash
# After #576-#582 (#566): allocation and GC time per step at 20 m, 72 threads, unpinned.
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --constraint=hwperf
#SBATCH -t 01:00:00
#SBATCH -o /home/atuin/b337bd/b337bd10/aeolus-bench/after581/memory_%j.out
W=/home/hpc/b337bd/b337bd10/smarttwin/Aeolus.jl/.worktrees
A=/home/atuin/b337bd/b337bd10/aeolus-bench
H=$W/issue-566/benchmarks/hpc
export BENCH_ENV=$A/env.sh BENCH_ROOT=$W BENCH_HPC=$H BENCH_BUILD=master-581
source $A/env.sh
P=$W/bench-master-581
echo "[$(date -Is)] $(hostname), bench/master-581 $(git -C $P rev-parse --short HEAD), scripts $(git -C $H rev-parse --short HEAD)"
(cd $P && julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()") || exit 1
bash $A/after581/guard.sh & GUARD=$!
unset JULIA_EXCLUSIVE
OUT=$A/results/after_581/memory; mkdir -p $OUT
julia -t 72,0 --project=$P $H/bench_run.jl memory $H/berlin_20m/berlin_20m_20250824.toml 0.4 $OUT/memory_20m_t72.toml \
    build=master-581 case=20m label=memory_20m_t72 nwarm=2 nmem=4 sample_rate=0.01 budget=900 || echo "FAILED"
cat $OUT/memory_20m_t72.memory.txt 2>/dev/null | head -40
kill $GUARD; echo "[$(date -Is)] done"
