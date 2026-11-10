#!/bin/bash
# In-loop sweep for the LinSolversGPU paper (Table 4 / Fig. 5), Chrono 10 branch gnsh/chrono10-cudss.
# ANCF 3443 shell panel, backward Euler, continuous integration, ten steps (tip load released after five).
# Pardiso (16 threads, iparm[7]=0) vs cuDSS 0.8 (library defaults); analysis at every call (cold/) and once (ao/).
# 9 grids, interleaved, 10 reps for grids 10-70 and 3 for 100-200. No CPU pinning.
# Each run starts only when no other GPU app runs and the load is low; completed logs are skipped on rerun.
# Usage: bash run_inloop_sweep.sh [build dir, default ../build10]
OUT="$(cd "$(dirname "$0")" && pwd)"
BUILD="${1:-$OUT/../build10}"
mkdir -p "$OUT/cold" "$OUT/ao"

others() { nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -v gnome-remote-desktop; }
wait_idle() {  # 60 s with no other GPU app and 1-min load below 1.5
  local idle=0
  while [ $idle -lt 6 ]; do
    if [ -z "$(others)" ] && awk "BEGIN{exit !($(cut -d' ' -f1 /proc/loadavg) < 1.5)}"; then idle=$((idle+1)); else idle=0; fi
    sleep 10
  done
}

for g in 10 20 30 40 50 70 100 140 200; do
  reps=10; [ $g -ge 100 ] && reps=3
  wait_idle
  for rep in $(seq 1 $reps); do
    for cfg in cold ao; do
      flag=""; [ $cfg = ao ] && flag="--analyze_once"
      for s in pardiso cudss; do
        bin="$BUILD/bin/jz_FEA_3443_check"; [ $s = cudss ] && bin="$BUILD/bin/jz_FEA_3443_check_cudss"
        log="$OUT/$cfg/g${g}_${s}_${rep}.log"
        grep -qs "Wall time (loop)" "$log" && continue
        /usr/bin/time -f "peakRSS=%MkB total=%es" timeout 2400 "$bin" --grid $g --nthreads 16 $flag > "$log" 2>&1
        echo "$(date +%T) grid$g $cfg $s rep$rep exit=$? $(grep -oE 'Wall time \(loop\) +: [0-9.]+' "$log")" | tee -a "$OUT/progress.txt"
      done
    done
  done
done
echo "$(date '+%F %T') DONE" | tee -a "$OUT/progress.txt"
