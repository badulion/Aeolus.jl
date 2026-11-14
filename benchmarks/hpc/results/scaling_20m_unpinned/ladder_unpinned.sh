#!/bin/bash
# 20 m scaling ladder after #572, unpinned (BENCH_PIN=0) (#566): scaling.sh 72 36 16 8 4 2 1 on one
# node, a /proc watcher alongside; then the pressure solve alone at 72 threads.
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --constraint=hwperf
#SBATCH -t 24:00:00
#SBATCH -o /home/atuin/b337bd/b337bd10/aeolus-bench/scaling20/ladder_unpinned_%j.out
W=/home/hpc/b337bd/b337bd10/smarttwin/Aeolus.jl/.worktrees
A=/home/atuin/b337bd/b337bd10/aeolus-bench
export BENCH_ENV=$A/env.sh BENCH_ROOT=$W BENCH_HPC=$W/issue-566/benchmarks/hpc
source $BENCH_ENV
OUT=$A/results/scaling_20m_unpinned; mkdir -p $OUT
echo "[$(date -Is)] $(hostname), bench-all $(git -C $W/bench-all rev-parse --short HEAD)"
(cd $W/bench-all && julia --project=. -e 'using Pkg; Pkg.instantiate()')
bash $A/scaling20/procwatch.sh > $OUT/procwatch.log 2>&1 & PW=$!
BENCH_PIN=0 bash $BENCH_HPC/scaling.sh $OUT 72 36 16 8 4 2 1
kill $PW
echo "[$(date -Is)] pressure solve alone"
julia -t 72,0 --project=$W/bench-all $BENCH_HPC/fft_20m/solver_compare.jl 1000 1000 150 72 estimate 3 2>&1 | tee $OUT/solver_compare_t72.txt
echo "[$(date -Is)] done"
