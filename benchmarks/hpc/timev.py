#!/usr/bin/env python3
# A stand-in for `/usr/bin/time -v -o <file> <command...>` (#566): Fritz's
# compute nodes have no GNU time (the login node does). Runs the command and
# writes, in GNU time's wording so the same greps work, its wall time, CPU
# time, exit status and maximum resident set size. The maximum RSS is that of
# the largest single process among the command and its waited-for
# descendants (getrusage RUSAGE_CHILDREN), as GNU time reports it, not the
# sum over processes running at once. Stdlib only.
#
#   python3 timev.py -o <file> <command...>
import os, resource, subprocess, sys, time

if len(sys.argv) < 4 or sys.argv[1] != "-o":
    sys.exit("usage: timev.py -o <file> <command...>")
out, cmd = sys.argv[2], sys.argv[3:]
t0 = time.time()
rc = subprocess.call(cmd)
wall = time.time() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
h, r = divmod(wall, 3600); m, s = divmod(r, 60)
with open(out, "w") as f:
    f.write(f'\tCommand being timed: "{" ".join(cmd)}"\n'
            f"\tUser time (seconds): {ru.ru_utime:.2f}\n"
            f"\tSystem time (seconds): {ru.ru_stime:.2f}\n"
            f"\tElapsed (wall clock) time (h:mm:ss or m:ss): {int(h)}:{int(m):02d}:{s:05.2f}\n"
            f"\tMaximum resident set size (kbytes): {ru.ru_maxrss}\n"   # Linux: kB
            f"\tExit status: {rc if rc >= 0 else 128 - rc}\n")
sys.exit(rc if rc >= 0 else 128 - rc)
