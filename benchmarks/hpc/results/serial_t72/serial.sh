#!/bin/bash
# Where the serial part of the step is (#566, after the unpinned ladder: main
# thread 78 % busy, the other 71 threads 37 %): bench_run.jl profile mode at
# -t 72,0, unpinned. Launch trace (serial time between launches, wall and busy
# per launch name) of 2 steps, then Profile of 3 steps split by thread and
# sleep state. 50 m first (quick, catches script errors), then 20 m.
#SBATCH -p singlenode
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -o /home/atuin/b337bd/b337bd10/aeolus-bench/scaling20/serial_%j.out
W=/home/hpc/b337bd/b337bd10/smarttwin/Aeolus.jl/.worktrees
A=/home/atuin/b337bd/b337bd10/aeolus-bench
H=$W/issue-566/benchmarks/hpc
source $A/env.sh
unset JULIA_EXCLUSIVE
OUT=$A/results/serial_t72; mkdir -p $OUT
echo "[$(date -Is)] $(hostname), bench-all $(git -C $W/bench-all rev-parse --short HEAD), scripts $(git -C $H rev-parse --short HEAD)"
julia -e 'Meta.parseall(read(ARGS[1], String)); println("bench_run.jl parses")' $H/bench_run.jl || exit 1
for c in "50m berlin_50m/berlin_50m_20250824.toml 1.0" "20m berlin_20m/berlin_20m_20250824.toml 0.4"; do
  set -- $c
  echo "[$(date +%T)] serial finder $1"
  julia -t 72,0 --project=$W/bench-all $H/bench_run.jl profile $H/$2 $3 $OUT/serial_$1_t72_all.toml \
      build=all case=$1 label=serial_$1_t72_all nwarm=2 nprof=3 ntrace=2 budget=900 || echo "FAILED $1"
  head -40 $OUT/serial_$1_t72_all.profile_threads.txt 2>/dev/null
done
echo "[$(date -Is)] done"
