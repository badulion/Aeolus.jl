#!/bin/bash
# After #576-#582 (#566): bench_run.jl profile at -t 72,0, unpinned, 50 m then 20 m
# (launch trace of 2 steps, Profile of 3 steps by thread). First the launch
# trace test (_planes3_kernel! traced), 2 workers.
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --constraint=hwperf
#SBATCH -t 01:30:00
#SBATCH -o /home/atuin/b337bd/b337bd10/aeolus-bench/after581/profile_%j.out
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
OUT=$A/results/after_581/profile; mkdir -p $OUT
(cd $P && julia --project=. test/run.jl --jobs 2 --only launch_trace.jl > $OUT/launch_trace_test.log 2>&1; echo "test exit $?" >> $OUT/launch_trace_test.log); tail -12 $OUT/launch_trace_test.log
for c in "50m berlin_50m/berlin_50m_20250824.toml 1.0" "20m berlin_20m/berlin_20m_20250824.toml 0.4"; do
  set -- $c
  echo "[$(date +%T)] profile $1"
  julia -t 72,0 --project=$P $H/bench_run.jl profile $H/$2 $3 $OUT/profile_$1_t72.toml \
      build=master-581 case=$1 label=profile_$1_t72 nwarm=2 nprof=3 ntrace=2 budget=900 || echo "FAILED $1"
  head -40 $OUT/profile_$1_t72.profile_threads.txt 2>/dev/null
done
kill $GUARD; echo "[$(date -Is)] done"
