#!/bin/bash
# In-loop ANCF shell (edge clamped by constraints), EULER_IMPLICIT, continuous integration (no --preint),
# Pardiso (16 threads) vs cuDSS, cold start and --analyze_once, 3 interleaved reps per grid.
# Logs: cold/g<grid>_<solver>_<rep>.log and ao/g<grid>_<solver>_<rep>.log (aggregate with ../jz_sweep_shell_ao/table4.py <dir>).
OUT="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT/../build" || exit 1
for g in 10 20 30 40 50 70 100 140 200; do
  for rep in 1 2 3; do
    for cfg in cold ao; do
      flag=""; [ $cfg = ao ] && flag="--analyze_once"
      for s in pardiso cudss; do
        bin=./bin/jz_FEA_3443_check; [ $s = cudss ] && bin=./bin/jz_FEA_3443_check_cudss
        log="$OUT/$cfg/g${g}_${s}_${rep}.log"
        /usr/bin/time -f "peakRSS=%MkB total=%es" timeout 2400 $bin --grid $g --nthreads 16 $flag > "$log" 2>&1
        echo "$(date +%T) grid$g $cfg $s rep$rep exit=$? $(grep -oE 'Wall time \(loop\) +: [0-9.]+' "$log") $(grep -o 'peakRSS=[0-9]*kB' "$log")"
      done
    done
  done
done
echo DONE
