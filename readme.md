# TIPs — Dental Inference API Server Setup Guide

Dental CBCT inference server for: **teeth segmentation, tooth numbering, pulp detection, mandible/maxilla structures**.

---

cd /home/oaiz/Documents/Sanora/dianexea_stack/TIPs
source ToothGroupNetwork/venv_tgn/bin/activate
python tooth_segmentation_server.py \
  --tgn_repo ./ToothGroupNetwork \
  --tgn_venv ./ToothGroupNetwork/venv_tgn \
  --port 7863


## System Requirements

- **OS**: Ubuntu 24.04
- **GPU**: NVIDIA with CUDA 13.0 support (tested on RTX 5060 Ti, SM_120 Blackwell)
- **Python**: 3.11
- **RAM**: 16GB recommended

---

## Step 1 — CUDA Toolkit (nvcc)

The driver alone is not enough. Install the CUDA compiler:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-nvcc-13-0
export PATH=/usr/local/cuda-13.0/bin:$PATH
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
nvcc --version  # should show 13.0
```

---

## Step 2 — Python Virtual Environment

The TIPs venv lives inside the repo at `./venv` so the project is self-contained.

```bash
cd /home/oaiz/Documents/Sanora/dianexea_stack/TIPs
python3.11 -m venv ./venv
source ./venv/bin/activate
```

---

## Step 3 — PyTorch (CUDA wheels)

> **Critical for Blackwell (SM_120) GPUs**: Standard PyTorch wheels do NOT support SM_120 until torch 2.11.0+.
> Install torch 2.11.0 from PyPI — it includes SM_120 support natively.

```bash
pip install torch torchvision
```

Verify SM_120 is supported:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list())"
# Should show 2.11.0+cu130 and include sm_120
```

---

## Step 4 — mamba-ssm + causal-conv1d (CUDA kernels)

> **Important**: both must be compiled from source against your installed torch.
> Pre-built wheels have ABI mismatches. The `--no-build-isolation` flag is required
> so the build uses your installed torch instead of downloading a new one.

### Step 4a — Patch CUDA 13.0 header (Ubuntu 25.10+ / 26.04, glibc 2.43+)

> **READ THIS BEFORE BUILDING.** On Ubuntu 26.04 (glibc 2.43) any `.cu` file fails with:
>
> ```
> error: exception specification is incompatible with that of previous function "rsqrt"
> ```
>
> This breaks **causal-conv1d, mamba-ssm, AND ToothGroupNetwork's pointops**.
>
> **Root cause:** glibc 2.43's `bits/mathcalls.h` declares `rsqrt`/`rsqrtf` with `noexcept(true)`, but CUDA 13.0's `crt/math_functions.h` declares them without `noexcept`. The two declarations conflict in C++. Switching GCC versions or `-ccbin` does NOT fix this — both compilers read the same conflicting headers. The fix is to patch the CUDA header itself (2 lines).

```bash
# One-time, machine-wide patch. Backup first so you can revert.
sudo cp /usr/local/cuda-13.0/include/crt/math_functions.h \
        /usr/local/cuda-13.0/include/crt/math_functions.h.bak
sudo sed -i 's/rsqrt(double x);/rsqrt(double x) noexcept(true);/' \
    /usr/local/cuda-13.0/include/crt/math_functions.h
sudo sed -i 's/rsqrtf(float x);/rsqrtf(float x) noexcept(true);/' \
    /usr/local/cuda-13.0/include/crt/math_functions.h

# Verify — both lines should now end with ` noexcept(true);`
grep -n "rsqrt[f]*(double x\|rsqrt[f]*(float x" /usr/local/cuda-13.0/include/crt/math_functions.h
```

This patch survives until you upgrade `cuda-nvcc-13-0` via apt (it'll overwrite the header — re-apply the sed if so).
A future CUDA 13.0.1+ / 13.1+ release should fix this upstream.

Then for each CUDA build, the standard env is enough — no GCC switch required:

```bash
source /home/oaiz/Documents/Sanora/dianexea_stack/TIPs/venv/bin/activate
export PATH=/usr/local/cuda-13.0/bin:$PATH
```

### Step 4b — causal-conv1d (~5 min)

```bash
MAX_JOBS=1 nice -n 19 pip install --no-cache-dir --no-build-isolation causal-conv1d
python -c "import causal_conv1d; print('causal_conv1d OK')"
```

### Step 4c — mamba-ssm from git (30–60 min, RAM/CPU heavy)

```bash
MAX_JOBS=1 nice -n 19 pip install --no-cache-dir --no-build-isolation git+https://github.com/state-spaces/mamba.git
python -c "from mamba_ssm import Mamba; print('mamba-ssm OK')"
```

Builds against torch 2.12+cu130 (or whatever was installed in Step 3).
**Let it complete fully.** If your desktop freezes, drop to TTY (`Ctrl+Alt+F3`) and watch with `htop` — do not reboot.

---

## Step 5 — Project Dependencies

```bash
pip install -r /path/to/TIPs/requirements.txt
```

---

## Step 6 — nnunetv2

Install official nnunetv2 2.6.4:

```bash
pip install nnunetv2==2.6.4
```

Copy TIPs custom trainers and nets into nnunetv2:

```bash
# Clone TIPs official repo (for trainer classes only)
cd /tmp
git clone https://github.com/TaoZhong11/TIPs tips_repo

NNUNET_DIR=$(python -c "import nnunetv2; import os; print(os.path.dirname(nnunetv2.__file__))")

# Copy custom network architectures
mkdir -p $NNUNET_DIR/nets
cp /tmp/tips_repo/nnunetv2/nets/*.py $NNUNET_DIR/nets/

# Copy custom trainers
cp /tmp/tips_repo/nnunetv2/training/nnUNetTrainer/nnUNetTrainer*.py $NNUNET_DIR/training/nnUNetTrainer/
cp /tmp/tips_repo/nnunetv2/training/nnUNetTrainer/nnUNetTrainerSwin*.py $NNUNET_DIR/training/nnUNetTrainer/
cp /tmp/tips_repo/nnunetv2/training/nnUNetTrainer/nnUNetTrainerSeg*.py $NNUNET_DIR/training/nnUNetTrainer/
```

Create the ToothSeg trainer stub (not in TIPs repo):

```bash
cat > $NNUNET_DIR/training/nnUNetTrainer/nnUNetTrainer_onlyMirror01_DASegOrd0.py << 'EOF'
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_onlyMirror01_DASegOrd0(nnUNetTrainer):
    """ToothSeg variant. Base trainer is sufficient for inference."""
    pass
EOF
```

---

## Step 7 — Model Weights

### ToothSeg weights (Dataset121 + Dataset123)
Download from: **https://zenodo.org/records/14893540**

```bash
mkdir -p /path/to/TIPs/nnResults
# Extract the downloaded zip/tar into nnResults/
# Result should be:
#   nnResults/Dataset121_ToothFairy2_Teeth/
#   nnResults/Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px/
```

### TIPs weights (Dataset810 pulp model)
Download from: **https://drive.google.com/file/d/1UuFgZ-kwRryPC-vK7w64xX0VO4iOAeGt/view**

```bash
pip install gdown
gdown 1UuFgZ-kwRryPC-vK7w64xX0VO4iOAeGt -O /tmp/tips_weights.zip
unzip /tmp/tips_weights.zip -d /path/to/TIPs/nnResults/
# Result should be:
#   nnResults/Dataset810_root_binarySDM/
```

### DentalSegmentator weights (Dataset112)
Already present in: `models/nnUNet/Dataset112_DentalSegmentator_v100/`

---

## Final Folder Structure

```
TIPs/
├── api_server_linux.py
├── requirements.txt
├── models/
│   └── nnUNet/
│       └── Dataset112_DentalSegmentator_v100/      ← dental model
├── nnResults/
│   ├── Dataset121_ToothFairy2_Teeth/               ← semseg (ToothSeg)
│   ├── Dataset123_ToothFairy2fixed_teeth_spacing02_brd3px/  ← instseg (ToothSeg)
│   └── Dataset810_root_binarySDM/                  ← pulp (TIPs UMambaBot)
├── toothseg/                                        ← postprocessing code
├── opg/best.pt                                      ← YOLO OPG model
└── t_number/best.pt                                 ← YOLO tooth numbering model
```

---

## Step 8 — Start the Server

```bash
cd /home/oaiz/Documents/Sanora/dianexea_stack/TIPs
source ./venv/bin/activate
# nnUNet env vars are set automatically by api_server_linux.py from REPO_ROOT
python api_server_linux.py

# Production launch (jemalloc + port override) — same command that start_server.sh / tips-server.service use:
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 PYTHONMALLOC=malloc python api_server_linux.py --port 7863
```

Expected startup output (all 4 models load in ~2-3 seconds):
```
[server] 1/4  semseg  Dataset121 ...
[server] 2/4  instseg Dataset123 ...
[server] 3/4  pulp    Dataset810 UMambaBot ...
[server] 4/4  dental  Dataset112 ...
[server] All models ready in 2.5s
[server] Worker ready — waiting for jobs
```

Server runs at: **http://0.0.0.0:7862**

---

## Known Warnings (all harmless)

| Warning | Cause | Status |
|---------|-------|--------|
| `pkg_resources is deprecated` | MONAI using old setuptools API | Harmless |
| `torch.cuda.amp.autocast deprecated` | UMambaBot nets use old autocast syntax | Harmless |
| `Detected old nnU-Net plans format` | Dataset810 trained on nnunetv2 2.1.1 | Handled automatically |

---

## Troubleshooting

### `nvcc not found`
```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
```

### `error: exception specification is incompatible with that of previous function "rsqrt"` (CUDA 13 + glibc 2.43)
Cost the maintainer **2 days** to figure out the first time. See Step 4a above for the full fix.
Short version — patch CUDA's `crt/math_functions.h` to add `noexcept(true)` to the `rsqrt`/`rsqrtf` declarations:
```bash
sudo cp /usr/local/cuda-13.0/include/crt/math_functions.h /usr/local/cuda-13.0/include/crt/math_functions.h.bak
sudo sed -i 's/rsqrt(double x);/rsqrt(double x) noexcept(true);/' /usr/local/cuda-13.0/include/crt/math_functions.h
sudo sed -i 's/rsqrtf(float x);/rsqrtf(float x) noexcept(true);/' /usr/local/cuda-13.0/include/crt/math_functions.h
# Re-apply if you ever apt upgrade cuda-nvcc-13-0 (it will overwrite the header).
```
**Note:** Switching GCC versions or setting `-ccbin` does NOT fix this — both compilers read the same conflicting glibc header. Only patching CUDA's header works.

### `mamba-ssm ABI mismatch / undefined symbol`
Means a pre-built wheel was used. Reinstall from source:
```bash
pip uninstall mamba-ssm -y
pip cache purge
MAX_JOBS=1 nice -n 19 pip install git+https://github.com/state-spaces/mamba.git --no-cache-dir --no-build-isolation
```

### `ModuleNotFoundError: No module named 'nnunetv2.nets'`
Copy the TIPs nets to nnunetv2:
```bash
NNUNET_DIR=$(python -c "import nnunetv2; import os; print(os.path.dirname(nnunetv2.__file__))")
mkdir -p $NNUNET_DIR/nets
cp /tmp/tips_repo/nnunetv2/nets/*.py $NNUNET_DIR/nets/
```

### System freezes during mamba-ssm build
Normal — CUDA compilation is RAM/CPU intensive. Use:
```bash
MAX_JOBS=1 nice -n 19 pip install ...
```
Do not restart. If desktop freezes press `Ctrl+Alt+F3` to open a TTY and check progress with `htop`.
