# ToothGroupNetwork — Python Inference Layer

## What it does
Labels each individual tooth from an intraoral scan (IOS) mesh with its FDI number (11-18, 21-28, 31-38, 41-48) + gingiva. Won MICCAI 3DTeethSeg'22 challenge. mIoU 90.16%.

## Hardware
- RTX 5060 Ti 16GB ✅ (needs 11GB+)
- PyTorch 2.6+ with CUDA 12.8 (Blackwell support)

## Setup

```bash
# 1. Clone
git clone https://github.com/limhoyeon/ToothGroupNetwork.git
cd ToothGroupNetwork

# 2. Install PyTorch for Blackwell (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install torch-geometric (match your torch + cuda version)
# Check matrix: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu128.html

# 4. Install other deps
pip install open3d trimesh numpy scikit-learn

# 5. Download pretrained weights
# Google Drive: https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby?usp=sharing
# Download ckpts(new).zip → unzip into ToothGroupNetwork/ckpts/
```

## Inference

```python
# Input: .obj mesh file (one arch at a time — upper or lower)
# Output: JSON with per-vertex tooth labels

python start_inference.py \
  --input_dir /path/to/obj/files \
  --output_dir /path/to/output \
  --model_type pointtransformer   # best accuracy
```

## FastAPI endpoint to add to api_server_linux.py

```python
import subprocess, json, tempfile, os
from fastapi import UploadFile, File

@app.post("/tooth-segmentation")
async def tooth_segmentation(file: UploadFile = File(...), arch: str = "upper"):
    """
    Input:  OBJ mesh file of one dental arch
    Output: JSON with per-vertex FDI tooth labels
    arch:   "upper" or "lower"
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded OBJ
        input_path = os.path.join(tmpdir, "input.obj")
        with open(input_path, "wb") as f:
            f.write(await file.read())

        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir)

        # Run inference
        result = subprocess.run([
            "python", "/path/to/ToothGroupNetwork/start_inference.py",
            "--input_dir", tmpdir,
            "--output_dir", output_dir,
            "--model_type", "pointtransformer",
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)

        # Read output JSON
        output_file = os.path.join(output_dir, "input.json")
        with open(output_file) as f:
            labels = json.load(f)

        return {"labels": labels, "arch": arch}
```

## NestJS integration (patient-aligners module)

When Phase 6 is implemented:
- Add `POST /patient-aligners/:id/segment-scan` endpoint in NestJS
- NestJS calls FastAPI `/tooth-segmentation` with the STL/OBJ file
- Returns labeled mesh to frontend
- Frontend renders colored teeth in existing STL viewer

## STL → OBJ conversion (if needed)

```python
import trimesh
mesh = trimesh.load("scan.stl")
mesh.export("scan.obj")
```

## Output format (Teeth3DS JSON)
```json
{
  "id_patient": "patient_001",
  "jaw": "upper",
  "labels": [11, 11, 12, 12, 0, 0, ...]  // per-vertex, 0 = gingiva, 11-48 = FDI tooth numbers
}
```

## Notes
- Process upper and lower arch separately (two separate API calls)
- Inference time: ~5-15 seconds per arch on RTX 5060 Ti
- Input must be IOS scan (intraoral scanner STL/OBJ) — NOT CBCT
- Compatible scanners: iTero, Medit, 3Shape Trios, Carestream (any that exports STL/OBJ)
