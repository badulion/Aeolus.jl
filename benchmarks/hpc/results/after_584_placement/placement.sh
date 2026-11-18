#!/bin/bash
# #584: the swap branch's unfused slowdown (+7..26 % at 36 threads, jobs 4474794,
# 4477000) - thread or page placement? bench_run.jl trace at -t 36,0, Berlin
# 20 m, nwarm=2 nsteps=4, builds master-581 (f6e8b10c) and 584-swap (27b12c34,
# with e918e33e), in three variants: unpinned (the 4477000 reference),
# pinned (JULIA_EXCLUSIVE=1), unpinned under numactl --interleave=all.
# numastat -p of each julia process every 20 s throughout.
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --constraint=hwperf
#SBATCH -t 02:00:00
#SBATCH -o /home/atuin/b337bd/b337bd10/aeolus-bench/placement/placement_%j.out
W=/home/hpc/b337bd/b337bd10/smarttwin/Aeolus.jl/.worktrees
A=/home/atuin/b337bd/b337bd10/aeolus-bench
H=$W/issue-566/benchmarks/hpc
source $A/env.sh
unset JULIA_EXCLUSIVE
OUT=$A/results/after_584_placement; mkdir -p $OUT
BUILDS="master-581 584-swap"
echo "[$(date -Is)] $(hostname), scripts $(git -C $H rev-parse --short HEAD)"
for b in $BUILDS; do echo "bench/$b $(git -C $W/bench-$b log -1 --format='%h %s')"; done
{ echo "hostname: $(hostname)"; echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing)"
  echo "--- numactl -H"; numactl -H; echo "--- lscpu | grep -i numa"; lscpu | grep -i numa; } > $OUT/node.txt
cat $OUT/node.txt
(cd $W/bench-584-swap && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' > $OUT/precompile_584-swap.log 2>&1) || { echo "precompile failed"; exit 1; }
bash $A/after584/guard.sh & GUARD=$!
run() {   # variant build prefix...
  local v=$1 b=$2; shift 2
  local lab=trace_20m_t36_${v}_$b
  echo "[$(date +%T)] $lab"
  ( while true; do for p in $(pgrep -x julia); do echo "=== $(date +%s.%N) pid $p"; numastat -p $p | tail -n +3; done; sleep 20; done ) > $OUT/numastat_$lab.log 2>&1 & local NS=$!
  "$@" julia -t 36,0 --project=$W/bench-$b $H/bench_run.jl trace $H/berlin_20m/berlin_20m_20250824.toml 0.4 $OUT/$lab.toml \
      build=$b case=20m label=$lab nwarm=2 nsteps=4 budget=1800 || echo "FAILED $lab"
  kill $NS
  grep -E "step_wall_s|launch_wall_s|serial_between_s|idle_in_launches_s|^thread_cpus" $OUT/$lab.toml
}
for b in $BUILDS; do run unpinned $b env; done
for b in $BUILDS; do run pinned $b env JULIA_EXCLUSIVE=1; done
for b in $BUILDS; do BENCH_NUMACTL="--interleave=all" run interleave $b numactl --interleave=all; done
kill $GUARD; echo "[$(date -Is)] done"
