#!/bin/bash
# ANCF 3443 shell, EULER_IMPLICIT, Pardiso (16 threads) vs cuDSS, 3 interleaved reps per resolution.
cd "$(dirname "$0")/../build" || exit 1
OUT="$(dirname "$0")"
for res in 0 2 4 8; do
  for rep in 1 2 3; do
    for s in pardiso cudss; do
      bin=./bin/jz_FEA_3443_check; [ $s = cudss ] && bin=./bin/jz_FEA_3443_check_cudss
      log="$OUT/res${res}_${s}_${rep}.log"
      /usr/bin/time -f "peakRSS=%MkB total=%es" timeout 1200 $bin --res$res --nthreads 16 --preint > "$log" 2>&1
      echo "$(date +%T) res$res $s rep$rep exit=$? $(grep -oE 'Wall time \(loop\) +: [0-9.]+' "$log") $(grep -o 'peakRSS=[0-9]*kB' "$log")"
    done
  done
done
echo DONE
