#!/bin/bash
# Single-shot build script for ToothGroupNetwork's pointops CUDA extension.
# Run with: bash /home/oaiz/Documents/Sanora/dianexea_stack/TIPs/build_pointops.sh

set -e

TGN_DIR="/home/oaiz/Documents/Sanora/dianexea_stack/TIPs/ToothGroupNetwork"
VENV_DIR="$TGN_DIR/venv_tgn"
POINTOPS_DIR="$TGN_DIR/external_libs/pointops"
LOG_FILE="/tmp/pointops_build.log"

# ── CUDA toolkit (auto-detects whichever cuda-X.Y is selected via update-alternatives) ──
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ── Sanity-check the CUDA / glibc 2.43 header patch ──
# On Ubuntu 25.10+ / 26.04 (glibc 2.43+), CUDA 12.8/13.0/13.1's crt/math_functions.h
# clashes with glibc's bits/mathcalls.h on rsqrt/rsqrtf noexcept declarations.
# Fix is a 2-line sed patch — see readme.md Step 4a.
# Refuses to build if the patch hasn't been applied, instead of letting nvcc
# fail 5 minutes into the compile.
MATH_H="$CUDA_HOME/include/crt/math_functions.h"
if ! grep -q 'rsqrt(double x) noexcept(true);' "$MATH_H" 2>/dev/null; then
    echo "ERROR: $MATH_H has not been patched for glibc 2.43+."
    echo "       Apply the rsqrt/rsqrtf noexcept patch first — see readme.md Step 4a."
    echo "       Quick fix:"
    echo "         sudo sed -i 's/rsqrt(double x);/rsqrt(double x) noexcept(true);/' $MATH_H"
    echo "         sudo sed -i 's/rsqrtf(float x);/rsqrtf(float x) noexcept(true);/' $MATH_H"
    exit 1
fi

# ── Force host compiler to gcc-13 ──
# Ubuntu 26.04's default g++ is GCC 15. CUDA 12.8's nvcc rejects host gcc > 14
# with "unsupported GNU version! gcc versions later than 14 are not supported".
# (Different bug from the rsqrt issue above — that one's a header conflict, this
# one's nvcc's own version check on the host compiler.)
if [ -x /usr/bin/g++-13 ]; then
    export CC=/usr/bin/gcc-13
    export CXX=/usr/bin/g++-13
    export CUDAHOSTCXX=/usr/bin/g++-13
else
    echo "ERROR: /usr/bin/g++-13 not found. Install with:  sudo apt install -y gcc-13 g++-13"
    exit 1
fi

# ── Add header paths for torch's CUDA backend (cusparse.h, cublas.h, cudnn.h, …) ──
# torch's CUDAContextLight.h includes <cusparse.h>; nvcc's system CUDA install (apt
# cuda-nvcc-12-8) doesn't ship cusparse/cublas/cudnn headers — only nvcc + runtime.
# The needed headers come bundled in the pip nvidia-* wheels inside this venv.
VENV_SITE="$VENV_DIR/lib/python3.11/site-packages"
NV_INCLUDES=""
for d in $VENV_SITE/nvidia/*/include; do
    [ -d "$d" ] && NV_INCLUDES="$NV_INCLUDES:$d"
done
NV_INCLUDES="${NV_INCLUDES#:}"  # strip leading colon
export CPATH="$NV_INCLUDES:${CPATH}"
export CPLUS_INCLUDE_PATH="$NV_INCLUDES:${CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="$NV_INCLUDES:${C_INCLUDE_PATH}"

# ── Target GPU: RTX 5060 Ti Blackwell sm_120 ──
export TORCH_CUDA_ARCH_LIST="12.0"

# ── Activate venv ──
source "$VENV_DIR/bin/activate"

echo "=========================================="
echo "Environment check:"
echo "=========================================="
echo "Python:       $(which python)"
echo "Torch:        $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA (torch): $(python -c 'import torch; print(torch.version.cuda)')"
echo "nvcc:         $(which nvcc)"
nvcc --version | tail -1
echo "Arch list:    $TORCH_CUDA_ARCH_LIST"
echo "=========================================="
echo ""

# ── Install ninja for faster parallel build (if missing) ──
python -c "import ninja" 2>/dev/null || pip install -q ninja

# ── Clean stale artifacts ──
cd "$POINTOPS_DIR"
echo "Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info
find . -maxdepth 1 -name "pointops_cuda*.so" -delete
echo ""

# ── Build ──
echo "Starting build — this may take 2-8 minutes..."
echo "Full log: $LOG_FILE"
echo ""

python setup.py install 2>&1 | tee "$LOG_FILE"
BUILD_EXIT=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $BUILD_EXIT -eq 0 ]; then
    echo "BUILD: SUCCESS"
    echo "Testing import..."
    python -c "import pointops_cuda; print('pointops_cuda imported OK')"
    IMPORT_EXIT=$?
    if [ $IMPORT_EXIT -eq 0 ]; then
        echo "IMPORT: SUCCESS"
        echo ""
        echo "READY. Next: run start_inference.py with a test OBJ."
    else
        echo "IMPORT: FAILED — built but cannot load. See errors above."
    fi
else
    echo "BUILD: FAILED with exit code $BUILD_EXIT"
    echo "Last 60 lines of log:"
    echo "------------------------------------------"
    tail -60 "$LOG_FILE"
fi
echo "=========================================="
