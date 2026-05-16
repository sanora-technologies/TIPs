# TIPs — Dental Inference API Server Setup Guide

Dental CBCT inference server for: **teeth segmentation, tooth numbering, pulp detection, mandible/maxilla structures**.

---

source /home/oaiz/Documents/sanora/Dianexea_stack/TIPs/ToothGroupNetwork/venv_tgn/bin/activate


cd /home/oaiz/Documents/sanora/Dianexea_stack/TIPs
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

```bash
python3.11 -m venv /home/oaiz/envs/server_env_3.11
source /home/oaiz/envs/server_env_3.11/bin/activate
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

## Step 4 — mamba-ssm (CUDA kernel)

> **Important**: mamba-ssm must be compiled from source against your installed torch.
> Pre-built wheels have ABI mismatches. The `--no-build-isolation` flag is required
> so the build uses your installed torch instead of downloading a new one.

```bash
MAX_JOBS=1 nice -n 19 pip install git+https://github.com/state-spaces/mamba.git --no-cache-dir --no-build-isolation
```

This will also upgrade torch to 2.11.0 and install triton, tilelang, quack-kernels.
**Let it complete fully — takes 30-60 min.**

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
source /home/oaiz/envs/server_env_3.11/bin/activate
export nnUNet_raw=/path/to/TIPs/nnUNet_raw
export nnUNet_preprocessed=/path/to/TIPs/nnUNet_preprocessed
export nnUNet_results=/path/to/TIPs/nnResults
cd /path/to/TIPs
python api_server_linux.py

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
