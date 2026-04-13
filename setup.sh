#!/bin/bash

ONEAPI_ROOT="/opt/intel/oneapi"

if [ -f "$ONEAPI_ROOT/setvars.sh" ]; then
    source "$ONEAPI_ROOT/setvars.sh" --force
    echo "Intel oneAPI environment initialized."
else
    echo "Error: setvars.sh not found at $ONEAPI_ROOT. Please check your installation path."
    exit 1
fi

# 3. Apply the AMD Performance Workaround (The "FakeIntel" Hack)
export MKL_DEBUG_CPU_TYPE=5

# 4. Optimize threading for AMD architecture
export MKL_NUM_THREADS=16
export MKL_DYNAMIC=FALSE

echo "Setup complete for AMD CPU."
echo "MKL_DEBUG_CPU_TYPE is set to: $MKL_DEBUG_CPU_TYPE"
echo "MKL_NUM_THREADS is set to: $MKL_NUM_THREADS"
