#!/bin/bash
# After #576-#582 (#566): 20 m unpinned scaling, 72 36 16 8 threads, /proc watcher alongside.
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --constraint=hwperf
#SBATCH -t 06:00:00
#SBATCH -o /home/atuin/b337bd/b337bd10/aeolus-bench/after581/ladder_unpinned_%j.out
W=/home/hpc/b337bd/b337bd10/smarttwin/Aeolus.jl/.worktrees
A=/home/atuin/b337bd/b337bd10/aeolus-bench
H=$W/issue-566/benchmarks/hpc
export BENCH_ENV=$A/env.sh BENCH_ROOT=$W BENCH_HPC=$H BENCH_BUILD=master-581
source $A/env.sh
P=$W/bench-master-581
echo "[$(date -Is)] $(hostname), bench/master-581 $(git -C $P rev-parse --short HEAD), scripts $(git -C $H rev-parse --short HEAD)"
(cd $P && julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()") || exit 1
bash $A/after581/guard.sh & GUARD=$!
OUT=$A/results/after_581/scaling_unpinned; mkdir -p $OUT
bash $A/scaling20/procwatch.sh > $OUT/procwatch.log 2>&1 & PW=$!
BENCH_PIN=0 bash $H/scaling.sh $OUT 72 36 16 8
kill $PW $GUARD; echo "[$(date -Is)] done"
