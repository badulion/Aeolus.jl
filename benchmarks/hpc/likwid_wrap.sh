#!/bin/bash
# likwid_wrap.sh <timeline-file> <command...>: run the command under
# likwid-perfctr in timeline mode (every 2 s, the groups of LIKWID_GROUPS
# multiplexed, all 72 cores), the timeline rows (stderr) into <timeline-file>.
# Needs `module load likwid` and a job with --constraint=hwperf (Fritz).
out=$1; shift
exec likwid-perfctr -c 0-71 ${LIKWID_GROUPS:--g MEM} -t 2s "$@" 2> "$out"
