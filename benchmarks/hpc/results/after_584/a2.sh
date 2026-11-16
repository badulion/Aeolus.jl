#!/bin/bash
# #584 (#581 A2, the step-start copy) on Fritz (#566): master (bench/master-581)
# against bench/584-swap (stage 1 swaps the state with the workspace, sweep out
# of place) and bench/584-copy (copy_storage! in chunks of >= 1 M points, one
# copyto! per task). Berlin 20 m, unpinned. First the three system images (in
# parallel, alongside precompiling each worktree); then per build, one after
# another: bench_run.jl profile at -t 72,0 (trace of 2 steps, then 10 profiled
# steps) and a plain time run at -t 36,0 (10 steps per setting).
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --constraint=hwperf
#SBATCH -t 03:00:00
#SBATCH -o /home/atuin/b337bd/b337bd10/aeolus-bench/after584/a2_%j.out
W=/home/hpc/b337bd/b337bd10/smarttwin/Aeolus.jl/.worktrees
A=/home/atuin/b337bd/b337bd10/aeolus-bench
H=$W/issue-566/benchmarks/hpc
source $A/env.sh
unset JULIA_EXCLUSIVE
export SYSIMG_CFG=$H/berlin_100m/berlin_100m_20250824.toml   # (added after job 4474794, whose image builds failed without it)
OUT=$A/results/after_584; mkdir -p $OUT $A/sysimages
BUILDS="master-581 584-swap 584-copy"
echo "[$(date -Is)] $(hostname), scripts $(git -C $H rev-parse --short HEAD)"
for b in $BUILDS; do echo "bench/$b $(git -C $W/bench-$b log -1 --format='%h %s')"; done
julia -e 'Meta.parseall(read(ARGS[1], String)); println("bench_run.jl parses")' $H/bench_run.jl || exit 1
bash $A/after584/guard.sh & GUARD=$!
for b in $BUILDS; do
  (cd $W/bench-$b && julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' > $OUT/precompile_$b.log 2>&1
   cd $H/sysimage && JULIA_LOAD_PATH="@:$H/sysimage:@stdlib" julia --project=. build.jl $W/bench-$b $A/sysimages/$b.so $A/sysimg_env/$b > $A/sysimages/$b.log 2>&1
   echo "sysimage $b exit $?" >> $A/sysimages/$b.log) &
done
wait $(jobs -p | grep -v "^$GUARD$")
for b in $BUILDS; do tail -2 $A/sysimages/$b.log; done
echo "[$(date -Is)] builds done"
for b in $BUILDS; do
  echo "[$(date +%T)] $b profile t72"
  julia -t 72,0 --project=$W/bench-$b $H/bench_run.jl profile $H/berlin_20m/berlin_20m_20250824.toml 0.4 $OUT/profile_20m_t72_$b.toml \
      build=$b case=20m label=profile_20m_t72_$b nwarm=2 ntrace=2 nprof=10 budget=1800 || echo "FAILED profile $b"
  grep -A2 "^\[profile\]" $OUT/profile_20m_t72_$b.toml; grep "alloc_gb_per_step\|^steps" $OUT/profile_20m_t72_$b.toml
  echo "[$(date +%T)] $b time t36"
  julia -t 36,0 --project=$W/bench-$b $H/bench_run.jl time $H/berlin_20m/berlin_20m_20250824.toml 0.4 $OUT/time_20m_t36_$b.toml \
      build=$b case=20m label=time_20m_t36_$b nwarm=2 rounds=1 nsteps=10 budget=1800 || echo "FAILED time $b"
  grep "step_median\|alloc_bytes\|gc_s" $OUT/time_20m_t36_$b.toml
done
kill $GUARD; echo "[$(date -Is)] done"
