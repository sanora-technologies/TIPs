#!/bin/bash
# Single-shot build script for ToothGroupNetwork's pointops CUDA extension.
# Run with: bash /home/oaiz/Documents/sanora/Dianexea_stack/TIPs/build_pointops.sh

set -e

TGN_DIR="/home/oaiz/Documents/sanora/Dianexea_stack/TIPs/ToothGroupNetwork"
VENV_DIR="$TGN_DIR/venv_tgn"
POINTOPS_DIR="$TGN_DIR/external_libs/pointops"
LOG_FILE="/tmp/pointops_build.log"

# ── CUDA toolkit ──
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ── Add bundled CUDA headers from pip nvidia-* packages (system CUDA is missing them) ──
VENV_SITE="$VENV_DIR/lib/python3.11/site-packages"
NV_INCLUDES="$VENV_SITE/nvidia/cu13/include:$VENV_SITE/nvidia/cudnn/include:$VENV_SITE/nvidia/cublas/include:$VENV_SITE/nvidia/cusparse/include:$VENV_SITE/nvidia/cusolver/include:$VENV_SITE/nvidia/cufft/include:$VENV_SITE/nvidia/curand/include:$VENV_SITE/nvidia/nccl/include"
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
