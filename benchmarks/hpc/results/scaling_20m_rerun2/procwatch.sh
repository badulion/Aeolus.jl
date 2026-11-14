#!/bin/bash
# /proc deltas over 10 s of the newest julia process, every 30 s (#566 rerun).
# Columns: time pid rss_GB busy_cores kernel_pct ctxsw_per_s nthreads
hz=$(getconf CLK_TCK)
snap() { local p=$1; read -r ut st < <(awk '{print $14, $15}' /proc/$p/stat 2>/dev/null)
  cs=$(cat /proc/$p/task/*/status 2>/dev/null | awk '/^(non)?voluntary_ctxt_switches/{s+=$2} END{print s+0}')
  echo "$ut $st $cs"; }
while true; do
  p=$(pgrep -n -x julia) || { sleep 30; continue; }
  read -r u1 s1 c1 < <(snap $p); sleep 10; read -r u2 s2 c2 < <(snap $p) || continue
  [ -z "$u2" ] && continue
  rss=$(awk '/VmRSS/{printf "%.1f", $2/2^20}' /proc/$p/status 2>/dev/null)
  nt=$(ls /proc/$p/task 2>/dev/null | wc -l)
  awk -v t="$(date +%T)" -v p=$p -v r=$rss -v hz=$hz -v du=$((u2-u1)) -v ds=$((s2-s1)) -v dc=$((c2-c1)) -v nt=$nt \
    'BEGIN{b=(du+ds)/hz/10; k=(du+ds)>0?100*ds/(du+ds):0; printf "%s %s %s GB  %.2f cores  %.0f%% kernel  %.0f ctxsw/s  %d threads\n", t,p,r,b,k,dc/10,nt}'
  sleep 20
done
