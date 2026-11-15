#!/bin/bash
# The stop rule of the 20 m handoff (#566): kill julia if its RSS exceeds 230 GB.
while true; do
  for p in $(pgrep -x julia); do
    r=$(awk '/VmRSS/{print int($2/2^20)}' /proc/$p/status 2>/dev/null)
    [ -n "$r" ] && [ "$r" -gt 230 ] && { echo "[$(date +%T)] STOP RULE: pid $p RSS ${r} GB > 230, killing"; kill $p; }
  done
  sleep 15
done
