#!/bin/bash
# In-loop ANCF shell (edge clamped by constraints), EULER_IMPLICIT, --preint, Pardiso (16 threads) vs cuDSS,
# 3 interleaved reps for the additional grids.
cd "$(dirname "$0")/../build" || exit 1
OUT="$(dirname "$0")"
for g in 30 40 70; do
  for rep in 1 2 3; do
    for s in pardiso cudss; do
      bin=./bin/jz_FEA_3443_check; [ $s = cudss ] && bin=./bin/jz_FEA_3443_check_cudss
      log="$OUT/g${g}_${s}_${rep}.log"
      /usr/bin/time -f "peakRSS=%MkB total=%es" timeout 1200 $bin --grid $g --nthreads 16 --preint > "$log" 2>&1
      echo "$(date +%T) grid$g $s rep$rep exit=$? $(grep -oE 'Wall time \(loop\) +: [0-9.]+' "$log") $(grep -o 'peakRSS=[0-9]*kB' "$log")"
    done
  done
done
echo DONE
