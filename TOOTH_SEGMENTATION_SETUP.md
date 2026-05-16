# Tooth Segmentation — Setup Guide (ToothGroupNetwork)

Production tooth segmentation from intraoral scan (IOS) meshes — STL or OBJ.

| Model | Accuracy | VRAM | Weights |
|---|---|---|---|
| **ToothGroupNetwork** | mIoU 90.16% (MICCAI 2022 winner) | 11 GB+ | Google Drive (manual download) |

Input must be an **IOS mesh** — intraoral scanner output (iTero, Medit, 3Shape Trios, Carestream).
**Not CBCT.** Not OPG. The patient puts the scanner in their mouth; you get an STL file.

---

## 1 — Prerequisites

```bash
nvidia-smi | grep "CUDA Version"
```

- **RTX 5060 Ti (Blackwell)** → CUDA 12.8 + PyTorch 2.6+
- **Older RTX (3090/4090)** → CUDA 12.1 + PyTorch 2.x

---

## 2 — Clone and install

```bash
cd /home/oaiz/Documents/sanora/Dianexea_stack/TIPs
git clone https://github.com/limhoyeon/ToothGroupNetwork.git
cd ToothGroupNetwork

# Dedicated venv (isolated from the main server)
python3 -m venv venv_tgn
source venv_tgn/bin/activate

# PyTorch — RTX 5060 Ti (Blackwell, CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# If your GPU is older (CUDA 12.1):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# torch-geometric — wheels must match your torch + cuda version
# Reference: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu128.html

# Other deps
pip install open3d trimesh numpy scikit-learn fastapi uvicorn python-multipart
```

---

## 3 — Download pretrained weights

1. Open: `https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby?usp=sharing`
2. Download `ckpts(new).zip`
3. Unzip into `ToothGroupNetwork/ckpts/`

After unzipping the structure should look like:
```
ToothGroupNetwork/
  ckpts/
    pointtransformer/
      best_model.pth   ← main weights (the one we use)
    dgcnn/
      best_model.pth
    pointnet2/
      best_model.pth
```

---

## 4 — Test inference manually

```bash
cd ToothGroupNetwork
source venv_tgn/bin/activate

mkdir -p test_input test_output
# Put a real .obj file in test_input/

python start_inference.py \
  --input_dir test_input \
  --output_dir test_output \
  --model_type pointtransformer

# Output:  test_output/<filename>.json
# Format:  {"labels": [11, 11, 12, 0, 0, 21, ...]}  ← per-vertex, 0=gingiva, FDI numbers
```

Expected time: 5–15 seconds per arch on RTX 5060 Ti.

---

## 5 — STL → OBJ conversion

ToothGroupNetwork requires `.obj`. The standalone server handles this automatically using `trimesh`, but here it is manually:

```python
import trimesh
mesh = trimesh.load("scan.stl")
mesh.export("scan.obj")
```

---

## 6 — Running the standalone segmentation server

The server lives at:
```
TIPs/tooth_segmentation_server.py
```

It runs on port **7863** (the main inference server runs on 7862).

```bash
cd /home/oaiz/Documents/sanora/Dianexea_stack/TIPs

python tooth_segmentation_server.py \
  --tgn_repo ./ToothGroupNetwork \
  --tgn_venv ./ToothGroupNetwork/venv_tgn \
  --port 7863
```

Test it:
```bash
curl -X POST http://localhost:7863/tooth-segmentation \
  -H "Authorization: Bearer sanora-webhook-secret" \
  -F "file=@/path/to/scan.obj" \
  -F "arch=upper"
```

Health check:
```bash
curl http://localhost:7863/health
```

---

## 7 — Integrating into api_server_linux.py

When you're ready to merge, look for `INTEGRATION POINT` comments inside
`tooth_segmentation_server.py`. There are 4 of them. Each labels exactly what
to copy and where it goes in the main server file.

You only need to:
1. Copy the **imports block**
2. Copy the **`ToothSegRunner` class**
3. Copy the **`/tooth-segmentation` route**
4. Add `tooth_seg = ToothSegRunner(...)` to the existing `startup()` function

No other changes to `api_server_linux.py` are required.

---

## 8 — Response format

```json
{
  "job_id": "uuid-here",
  "arch": "upper",
  "vertex_count": 18432,
  "labels": [11, 11, 12, 12, 0, 0, 21, 21, ...],
  "label_summary": {
    "0":  1204,
    "11": 843,
    "12": 912,
    "13": 701
  },
  "model_used": "ToothGroupNetwork/pointtransformer",
  "inference_seconds": 8.3
}
```

`labels` is a flat array — one integer per vertex.

| Range | Meaning |
|---|---|
| `0` | Gingiva / unlabeled |
| `11–18` | Upper right (UR8 .. UR1) |
| `21–28` | Upper left  (UL1 .. UL8) |
| `31–38` | Lower left  (LL1 .. LL8) |
| `41–48` | Lower right (LR8 .. LR1) |

---

## 9 — Notes

- Process upper and lower arch **separately** — one API call per arch
- Input must be IOS mesh — NOT CBCT, NOT OPG
- The server uses the same `AI_WEBHOOK_SECRET` bearer token as the main inference server
- Inference runs in a **subprocess** using the dedicated `venv_tgn` interpreter, so a crash or OOM inside the model does not kill the API server
