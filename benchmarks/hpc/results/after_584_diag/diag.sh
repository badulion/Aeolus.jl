#!/bin/bash
# #584: why swap's unfused path (loop_fusion = never) is ~2 s/step slower than
# master at 36 threads (job 4474794: 9.65 vs 7.67 s). bench_run.jl trace at
# -t 36,0, Berlin 20 m, unpinned, master-581 then 584-swap: per setting 2
# discarded + 4 traced steps; launch wall, serial time between launches, and
# busy per launch name.
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --constraint=hwperf
#SBATCH -t 01:00:00
#SBATCH -o /home/atuin/b337bd/b337bd10/aeolus-bench/after584/diag_%j.out
W=/home/hpc/b337bd/b337bd10/smarttwin/Aeolus.jl/.worktrees
A=/home/atuin/b337bd/b337bd10/aeolus-bench
H=$W/issue-566/benchmarks/hpc
source $A/env.sh
unset JULIA_EXCLUSIVE
OUT=$A/results/after_584_diag; mkdir -p $OUT
echo "[$(date -Is)] $(hostname), scripts $(git -C $H rev-parse --short HEAD)"
for b in master-581 584-swap; do echo "bench/$b $(git -C $W/bench-$b log -1 --format='%h %s')"; done
bash $A/after584/guard.sh & GUARD=$!
for b in master-581 584-swap; do
  echo "[$(date +%T)] $b trace t36"
  julia -t 36,0 --project=$W/bench-$b $H/bench_run.jl trace $H/berlin_20m/berlin_20m_20250824.toml 0.4 $OUT/trace_20m_t36_$b.toml \
      build=$b case=20m label=trace_20m_t36_$b nwarm=2 nsteps=4 budget=1800 || echo "FAILED trace $b"
  grep -E "step_wall_s|launch_wall_s|serial_between_s|idle_in_launches_s" $OUT/trace_20m_t36_$b.toml
done
kill $GUARD; echo "[$(date -Is)] done"
