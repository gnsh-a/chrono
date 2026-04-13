#!/bin/bash

> output.txt
./setup.sh

for i in {1..5}; do
    echo "Standard run $i at res 0"
    echo "Standard run $i at res 0" >> output.txt
    ./build/bin/jz_FEA_3243_check --res 0 --nthreads 16 >> output.txt 2>&1
done

# Sub-loop 2: CUDSS Check (5 runs)
for i in {1..5}; do
    echo "CUDSS run $i at res 0"
    echo "CUDSS run $i at res 0s" >> output.txt
    ./build/bin/jz_FEA_3243_check_cudss --res 0 --nthreads 16 >> output.txt 2>&1
done

for ((res=2; res<=64; res*=2)); do
    echo "--- Starting Resolution: $res ---"
    echo "--- Starting Resolution: $res ---" >> output.txt

    # Sub-loop 1: Standard Check (5 runs)
    for i in {1..5}; do
        echo "Standard run $i at res $res"
        echo "Standard run $i at res $res" >> output.txt
        ./build/bin/jz_FEA_3243_check --res $res --nthreads 16 >> output.txt 2>&1
    done

    # Sub-loop 2: CUDSS Check (5 runs)
    for i in {1..5}; do
        echo "CUDSS run $i at res $res"
        echo "CUDSS run $i at res $res" >> output.txt
        ./build/bin/jz_FEA_3243_check_cudss --res $res --nthreads 16 >> output.txt 2>&1
    done
done

echo "Benchmarks complete. Check output.txt for details."
