"""
tooth_segmentation_server.py — Standalone FastAPI server for IOS mesh tooth segmentation.

Runs on port 7863 (main inference server is 7862).
Backend: ToothGroupNetwork (MICCAI 2022 winner, pointtransformer variant).

Usage:
    python tooth_segmentation_server.py \
        --tgn_repo ./ToothGroupNetwork \
        --tgn_venv ./ToothGroupNetwork/venv_tgn \
        --port 7863

Test:
    curl -X POST http://localhost:7863/tooth-segmentation \
        -H "Authorization: Bearer sanora-webhook-secret" \
        -F "file=@scan.obj" \
        -F "arch=upper"

─────────────────────────────────────────────────────────────────────────────────
INTEGRATION INTO api_server_linux.py
─────────────────────────────────────────────────────────────────────────────────
When you're ready to fold this into the main server, follow the 4 steps tagged
with "INTEGRATION POINT" comments below.  Nothing else in api_server_linux.py
needs to change.
─────────────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINT 1 — IMPORTS
# Copy everything between the START/END markers to the top of api_server_linux.py
# (after the existing imports block, before the first class definition)
# ─── START ───────────────────────────────────────────────────────────────────
import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import trimesh  # pip install trimesh
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# ─── END ─────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tooth_seg")


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINT 2 — ToothSegRunner CLASS
# Copy this entire class into api_server_linux.py, right next to the other
# model-wrapper classes (wherever you keep CBCTModel, OPGModel, etc.)
# ─── START ───────────────────────────────────────────────────────────────────
class ToothSegRunner:
    """
    Runs ToothGroupNetwork inference in a subprocess so a crash or OOM
    inside the model does not kill the main API server.

    The subprocess uses ToothGroupNetwork's own venv python interpreter,
    so its dependencies (torch-geometric, torch-scatter, etc.) are fully
    isolated from this process's imports.
    """

    def __init__(
        self,
        tgn_repo: str = "./ToothGroupNetwork",
        tgn_venv: str = "./ToothGroupNetwork/venv_tgn",
        device: str = "cuda",
    ):
        self.tgn_repo = Path(tgn_repo).resolve()
        self.tgn_venv = Path(tgn_venv).resolve()
        self.device   = device
        self._validate_paths()
        log.info(f"ToothSegRunner ready — repo={self.tgn_repo} device={self.device}")

    def _validate_paths(self):
        if not self.tgn_repo.exists():
            raise FileNotFoundError(
                f"ToothGroupNetwork repo not found at {self.tgn_repo}. "
                "Clone it: git clone https://github.com/limhoyeon/ToothGroupNetwork.git"
            )
        ckpts_dir = self.tgn_repo / "ckpts"
        required = [ckpts_dir / "tgnet_fps.h5", ckpts_dir / "tgnet_bdl.h5"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Required weights missing: {missing}. "
                "Download ckpts(new).zip from "
                "https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby"
                " and move all .h5 files into ToothGroupNetwork/ckpts/"
            )

    def _python(self, venv_path: Path) -> str:
        """Return path to python interpreter in a venv."""
        for candidate in [
            venv_path / "bin" / "python",
            venv_path / "bin" / "python3",
            venv_path / "Scripts" / "python.exe",  # Windows fallback
        ]:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"No python interpreter found in venv at {venv_path}")

    def _stl_to_obj(self, stl_path: str, obj_path: str):
        """Convert STL → OBJ — ToothGroupNetwork requires OBJ input."""
        mesh = trimesh.load(stl_path)
        mesh.export(obj_path)

    def run(self, mesh_file: bytes, filename: str, arch: str = "upper") -> dict:
        """
        Main entry point.

        Args:
            mesh_file: raw bytes of the uploaded STL or OBJ file
            filename:  original filename (used to detect format)
            arch:      "upper" or "lower" (informational, passed through to response)

        Returns:
            dict with keys: labels, vertex_count, label_summary, model_used, inference_seconds
        """
        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ext = Path(filename).suffix.lower()

            uploaded = tmp / f"input{ext}"
            uploaded.write_bytes(mesh_file)

            if ext != ".obj":
                obj_path = tmp / "input.obj"
                self._stl_to_obj(str(uploaded), str(obj_path))
                input_file = obj_path
            else:
                input_file = uploaded

            self._current_arch = arch
            labels = self._run_tgn(tmp, input_file)

        elapsed = round(time.perf_counter() - t0, 2)

        label_summary: dict[str, int] = {}
        for lbl in labels:
            k = str(lbl)
            label_summary[k] = label_summary.get(k, 0) + 1

        return {
            "labels":            labels,
            "vertex_count":      len(labels),
            "label_summary":     label_summary,
            "model_used":        "ToothGroupNetwork/tgnet",
            "inference_seconds": elapsed,
        }

    def _run_tgn(self, tmp: Path, input_obj: Path) -> list[int]:
        """
        Calls ToothGroupNetwork's start_inference.py in a subprocess.
        Returns per-vertex FDI label list.

        TGN's expected layout (Teeth3DS-style):
          input_dir/
            <base_name>/          ← subdir whose name matches a line in split.txt
              <base_name>.obj
          split.txt               ← one base name per line
          checkpoint paths        ← without .h5; the script appends it
        """
        output_dir = tmp / "output"
        output_dir.mkdir()

        # TGN requires filename format: {casename}_{arch}.obj inside {casename}/ subfolder.
        # We use "scan" as the casename — full filename becomes scan_upper.obj or scan_lower.obj.
        casename     = "scan"
        full_name    = f"{casename}_{self._current_arch}"  # e.g. "scan_upper"
        input_dir    = tmp / "input"
        scan_subdir  = input_dir / casename
        scan_subdir.mkdir(parents=True)
        shutil.copy(str(input_obj), str(scan_subdir / f"{full_name}.obj"))

        # split.txt: TGN matches on subdir basename, which is the casename
        split_txt = tmp / "split.txt"
        split_txt.write_text(f"{casename}\n")

        python = self._python(self.tgn_venv)
        cmd = [
            python,
            str(self.tgn_repo / "start_inference.py"),
            "--input_dir_path",       str(input_dir),
            "--split_txt_path",       str(split_txt),
            "--save_path",            str(output_dir),
            "--model_name",           "tgnet",
            "--checkpoint_path",      "ckpts/tgnet_fps",   # no .h5 — script appends it
            "--checkpoint_path_bdl",  "ckpts/tgnet_bdl",   # no .h5 — script appends it
        ]

        # Subprocess needs torch's libc10.so + bundled CUDA libs on LD_LIBRARY_PATH
        site_pkgs = self.tgn_venv / "lib" / "python3.11" / "site-packages"
        torch_lib = site_pkgs / "torch" / "lib"
        nv_libs = [
            site_pkgs / "nvidia" / "cu13" / "lib",
            site_pkgs / "nvidia" / "cudnn" / "lib",
            site_pkgs / "nvidia" / "cublas" / "lib",
            site_pkgs / "nvidia" / "cusparse" / "lib",
            site_pkgs / "nvidia" / "cusolver" / "lib",
            site_pkgs / "nvidia" / "cufft" / "lib",
            site_pkgs / "nvidia" / "curand" / "lib",
            site_pkgs / "nvidia" / "nccl" / "lib",
        ]
        ld_paths = [str(torch_lib)] + [str(p) for p in nv_libs if p.exists()]
        ld_paths += ["/usr/local/cuda-13.0/lib64"]

        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0" if self.device == "cuda" else "",
            "LD_LIBRARY_PATH":      ":".join(ld_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
        }

        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(self.tgn_repo))
        if result.returncode != 0:
            log.error(f"TGN stderr:\n{result.stderr}")
            raise RuntimeError(f"ToothGroupNetwork inference failed: {result.stderr[-500:]}")

        # TGN names output after the obj filename: scan_upper.obj → scan_upper.json
        output_json = output_dir / f"{full_name}.json"
        if not output_json.exists():
            candidates = list(output_dir.glob("*.json"))
            if not candidates:
                raise RuntimeError("ToothGroupNetwork produced no output JSON")
            output_json = candidates[0]

        with open(output_json) as f:
            data = json.load(f)

        # Output may be a flat list, {"labels": [...]} or {"pred": [...]}
        if isinstance(data, list):
            return data
        return data.get("labels", data.get("pred", []))
# ─── END ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINT 3 — FASTAPI ROUTE
# Copy the route function (the one decorated with @app.post in the standalone
# section below) into api_server_linux.py's routes section.
# It must come AFTER `app = FastAPI(...)` and AFTER `tooth_seg` is assigned
# inside startup() (see INTEGRATION POINT 4).
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION POINT 4 — STARTUP
# Inside api_server_linux.py's @app.on_event("startup") or lifespan function,
# add these lines after the existing model loads:
#
#   global tooth_seg
#   tooth_seg = ToothSegRunner(
#       tgn_repo=args.tgn_repo,
#       tgn_venv=args.tgn_venv,
#       device=args.device,
#   )
#
# And add to argparse in __main__:
#   parser.add_argument("--tgn-repo", default="./ToothGroupNetwork")
#   parser.add_argument("--tgn-venv", default="./ToothGroupNetwork/venv_tgn")
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE SERVER — everything below runs only when this file is executed
# directly.  None of it is needed when integrating into api_server_linux.py.
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Standalone tooth segmentation server")
    p.add_argument("--tgn_repo", default="./ToothGroupNetwork",
                   help="Path to cloned ToothGroupNetwork repo")
    p.add_argument("--tgn_venv", default="./ToothGroupNetwork/venv_tgn",
                   help="Path to ToothGroupNetwork's venv")
    p.add_argument("--device",   default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--port",     default=7863, type=int)
    p.add_argument("--secret",   default=None,
                   help="Bearer token (falls back to AI_WEBHOOK_SECRET env var, "
                        "then 'sanora-webhook-secret')")
    return p.parse_args()


args = _parse_args()

# ── Auth (same pattern as api_server_linux.py) ────────────────────────────────
_WEBHOOK_SECRET = (
    args.secret
    or os.environ.get("AI_WEBHOOK_SECRET", "sanora-webhook-secret")
)
_bearer = HTTPBearer()

def _verify_token(creds: HTTPAuthorizationCredentials = Depends(_bearer)):
    if creds.credentials != _WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Tooth Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load runner at startup ────────────────────────────────────────────────────
tooth_seg: ToothSegRunner | None = None

@app.on_event("startup")
async def startup():
    global tooth_seg
    try:
        tooth_seg = ToothSegRunner(
            tgn_repo = args.tgn_repo,
            tgn_venv = args.tgn_venv,
            device   = args.device,
        )
    except FileNotFoundError as e:
        # Log but don't crash — server still starts, endpoint returns 503
        log.error(f"ToothSegRunner init failed: {e}")
        tooth_seg = None


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok" if tooth_seg else "degraded",
        "device": tooth_seg.device if tooth_seg else None,
    }


# ── Minimal upload UI ─────────────────────────────────────────────────────────
_UI_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Tooth Segmentation</title>
  <style>
    *{box-sizing:border-box;font-family:-apple-system,system-ui,sans-serif}
    body{margin:0;background:#0f172a;color:#e2e8f0;min-height:100vh;padding:24px}
    h1{margin:0 0 4px;font-size:22px;font-weight:600;color:#fff}
    .sub{color:#94a3b8;font-size:13px;margin-bottom:24px}
    .card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px;margin-bottom:16px;max-width:1100px}
    .row{display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
    label{font-size:12px;color:#94a3b8;display:block;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em}
    input[type=file],input[type=text],select{background:#0f172a;color:#e2e8f0;border:1px solid #475569;border-radius:8px;padding:8px 12px;font-size:14px;width:100%}
    input[type=file]{padding:6px;cursor:pointer}
    button{background:#6366f1;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:14px;font-weight:600;cursor:pointer;transition:background .15s}
    button:hover:not(:disabled){background:#4f46e5}
    button:disabled{background:#475569;cursor:not-allowed}
    .field{flex:1;min-width:180px}
    .status{padding:10px 14px;border-radius:8px;font-size:13px;margin-top:12px;display:none}
    .status.show{display:block}
    .status.loading{background:#1e3a8a;color:#bfdbfe}
    .status.error{background:#7f1d1d;color:#fecaca}
    .status.ok{background:#14532d;color:#bbf7d0}
    .meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-bottom:16px}
    .meta-item{background:#0f172a;border-radius:8px;padding:10px}
    .meta-label{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
    .meta-value{font-size:16px;font-weight:600;color:#fff;margin-top:2px}
    .legend{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:6px;margin-bottom:16px}
    .tooth{display:flex;align-items:center;gap:8px;background:#0f172a;border-radius:6px;padding:6px 10px;font-size:12px}
    .swatch{width:14px;height:14px;border-radius:3px;flex-shrink:0}
    .tooth-id{font-weight:600;color:#fff;min-width:24px}
    .tooth-count{color:#94a3b8;font-size:11px;margin-left:auto}
    pre{background:#0f172a;border-radius:8px;padding:12px;overflow:auto;font-size:11px;color:#94a3b8;max-height:200px}
    #viewer{width:100%;height:520px;background:#020617;border-radius:8px;position:relative;overflow:hidden}
    #viewer-overlay{position:absolute;top:10px;left:10px;background:rgba(15,23,42,.8);padding:6px 10px;border-radius:6px;font-size:11px;color:#94a3b8;pointer-events:none}
    #tooltip{position:fixed;pointer-events:none;background:#1e293b;border:1px solid #475569;color:#fff;padding:5px 9px;border-radius:6px;font-size:12px;font-weight:600;display:none;z-index:1000;box-shadow:0 4px 12px rgba(0,0,0,.4)}
    #tooltip .tip-sub{display:block;font-size:10px;font-weight:400;color:#94a3b8;margin-top:1px}
    .tooth{cursor:pointer;transition:transform .1s,background .15s}
    .tooth:hover{background:#1e293b}
    .tooth.selected{outline:2px solid #818cf8;background:#1e293b}
    .viewer-controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;font-size:12px}
    .viewer-controls button{padding:5px 10px;font-size:11px;background:#334155}
    .viewer-controls button:hover:not(:disabled){background:#475569}
    .viewer-controls button.active{background:#6366f1}
  </style>
</head>
<body>
  <h1>Tooth Segmentation</h1>
  <div class="sub">ToothGroupNetwork inference — upload an intraoral scan to get per-vertex FDI tooth labels.</div>

  <div class="card">
    <div class="row">
      <div class="field">
        <label>Mesh file (.obj only for 3D view)</label>
        <input type="file" id="file" accept=".obj,.stl">
      </div>
      <div class="field" style="max-width:140px">
        <label>Arch</label>
        <select id="arch">
          <option value="upper">Upper</option>
          <option value="lower">Lower</option>
        </select>
      </div>
      <div class="field">
        <label>Bearer token</label>
        <input type="text" id="token" value="sanora-webhook-secret">
      </div>
      <div style="align-self:flex-end">
        <button id="submit">Segment</button>
      </div>
    </div>
    <div id="status" class="status"></div>
  </div>

  <div id="tooltip"></div>

  <div class="card" id="result" style="display:none">
    <div class="meta" id="meta"></div>

    <label>3D segmentation view</label>
    <div id="viewer">
      <div id="viewer-overlay">Drag to rotate · Scroll to zoom · Right-drag to pan</div>
    </div>
    <div class="viewer-controls">
      <button id="btn-solid"  class="active">Solid</button>
      <button id="btn-wire">Wireframe</button>
      <button id="btn-hide-gum">Hide gum</button>
      <button id="btn-reset-view">Reset view</button>
    </div>

    <div style="margin-top:20px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center">
      <label style="margin:0">Tooth label summary</label>
      <span style="font-size:10px;color:#64748b">Click a tooth to isolate · click again to clear</span>
    </div>
    <div class="legend" id="legend"></div>

    <details style="margin-top:16px">
      <summary style="cursor:pointer;color:#94a3b8;font-size:12px">Raw JSON (labels array truncated)</summary>
      <pre id="raw"></pre>
    </details>
  </div>

<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── FDI color palette ──
const FDI_COLORS = {0:'#475569'};
const PAL_R = ['#dc2626','#ea580c','#d97706','#ca8a04','#65a30d','#16a34a','#0891b2','#2563eb'];
const PAL_L = ['#7c3aed','#c026d3','#db2777','#e11d48','#fb7185','#f87171','#fbbf24','#facc15'];
for(let i=1;i<=8;i++){
  FDI_COLORS[10+i] = PAL_R[i-1]; FDI_COLORS[20+i] = PAL_L[i-1];
  FDI_COLORS[30+i] = PAL_L[i-1]; FDI_COLORS[40+i] = PAL_R[i-1];
}
const hexToRGB = h => {
  const n = parseInt(h.slice(1), 16);
  return [((n>>16)&255)/255, ((n>>8)&255)/255, (n&255)/255];
};

const $ = id => document.getElementById(id);
const setStatus = (msg, kind='loading') => {
  const s = $('status'); s.className = 'status show ' + kind; s.textContent = msg;
};

// ── OBJ parser — vertices + faces in original order (matches TGN label order) ──
function parseOBJ(text){
  const verts = [];
  const faces = [];
  const lines = text.split(/\r?\n/);
  for(const line of lines){
    if(line.startsWith('v ')){
      const p = line.split(/\s+/);
      verts.push(+p[1], +p[2], +p[3]);
    } else if(line.startsWith('f ')){
      const p = line.split(/\s+/);
      // OBJ faces are 1-indexed; handle "v", "v/vt", "v/vt/vn", "v//vn"
      const idx = p.slice(1).map(t => parseInt(t.split('/')[0], 10) - 1);
      // Triangulate fan if quad/ngon
      for(let i=1; i<idx.length-1; i++){
        faces.push(idx[0], idx[i], idx[i+1]);
      }
    }
  }
  return { verts, faces };
}

// ── Three.js setup ──
let scene, camera, renderer, controls, mesh, wireframe;
let initialCamPos = null, initialTarget = null;
let selectedTooth = null;   // FDI int or null
let gumHidden    = false;
let raycaster, mouseNDC;

function initViewer(){
  const container = $('viewer');
  const w = container.clientWidth, h = container.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x020617);

  camera = new THREE.PerspectiveCamera(45, w/h, 0.1, 5000);
  camera.position.set(0, 0, 150);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h);
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dir1 = new THREE.DirectionalLight(0xffffff, 0.9); dir1.position.set(1, 1, 1); scene.add(dir1);
  const dir2 = new THREE.DirectionalLight(0xffffff, 0.4); dir2.position.set(-1, -1, -1); scene.add(dir2);

  raycaster = new THREE.Raycaster();
  mouseNDC  = new THREE.Vector2();

  container.addEventListener('pointermove', onHover);
  container.addEventListener('pointerleave', () => $('tooltip').style.display = 'none');

  window.addEventListener('resize', () => {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w/h; camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  (function animate(){
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();
}

function onHover(e){
  if(!mesh || !window._lastLabels) return;
  const container = $('viewer');
  const rect = container.getBoundingClientRect();
  mouseNDC.x = ((e.clientX - rect.left) / rect.width)  *  2 - 1;
  mouseNDC.y = ((e.clientY - rect.top)  / rect.height) * -2 + 1;
  raycaster.setFromCamera(mouseNDC, camera);
  const hits = raycaster.intersectObject(mesh);
  const tip = $('tooltip');
  if(hits.length){
    const vIdx = hits[0].face.a;
    const lbl  = window._lastLabels[vIdx];
    const name = (lbl === 0) ? 'Gingiva' : `FDI ${lbl}`;
    const cnt  = window._lastSummary[lbl] || 0;
    tip.innerHTML = `${name}<span class="tip-sub">${cnt.toLocaleString()} vertices</span>`;
    tip.style.left = (e.clientX + 12) + 'px';
    tip.style.top  = (e.clientY + 12) + 'px';
    tip.style.display = 'block';
  } else {
    tip.style.display = 'none';
  }
}

function renderMesh(verts, faces, labels){
  if(mesh){ scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); }
  if(wireframe){ scene.remove(wireframe); wireframe.geometry.dispose(); wireframe.material.dispose(); }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  geom.setIndex(faces);
  geom.computeVertexNormals();

  // Per-vertex colors based on labels
  const colors = new Float32Array(labels.length * 3);
  for(let i=0; i<labels.length; i++){
    const c = hexToRGB(FDI_COLORS[labels[i]] || '#64748b');
    colors[i*3]     = c[0];
    colors[i*3 + 1] = c[1];
    colors[i*3 + 2] = c[2];
  }
  geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  // Center + scale
  geom.computeBoundingBox();
  const box = geom.boundingBox;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  geom.translate(-center.x, -center.y, -center.z);

  const mat = new THREE.MeshPhongMaterial({
    vertexColors: true, flatShading: false, shininess: 25, side: THREE.DoubleSide,
  });
  mesh = new THREE.Mesh(geom, mat);
  scene.add(mesh);

  // Wireframe (hidden initially)
  wireframe = new THREE.LineSegments(
    new THREE.WireframeGeometry(geom),
    new THREE.LineBasicMaterial({ color: 0x6366f1, transparent: true, opacity: 0.3 })
  );
  wireframe.visible = false;
  scene.add(wireframe);

  // Frame the camera
  camera.position.set(0, size * 0.4, size * 1.4);
  camera.lookAt(0, 0, 0);
  initialCamPos = camera.position.clone();
  initialTarget = new THREE.Vector3(0,0,0);
  controls.target.copy(initialTarget);
  controls.update();
}

// ── Recompute vertex colors based on (selectedTooth, gumHidden) state ──
function recolor(){
  if(!mesh || !window._lastLabels) return;
  const labels = window._lastLabels;
  const colors = mesh.geometry.attributes.color.array;
  const dim = [0.13, 0.16, 0.21];                   // dark slate for non-selected
  const gumDimColor = [0.05, 0.06, 0.09];           // near-background for hidden gum
  for(let i=0; i<labels.length; i++){
    const lbl = labels[i];
    let c;
    if(gumHidden && lbl === 0){
      c = gumDimColor;
    } else if(selectedTooth !== null && lbl !== selectedTooth){
      c = dim;
    } else {
      c = hexToRGB(FDI_COLORS[lbl] || '#64748b');
    }
    colors[i*3]     = c[0];
    colors[i*3 + 1] = c[1];
    colors[i*3 + 2] = c[2];
  }
  mesh.geometry.attributes.color.needsUpdate = true;
}

// ── Camera fly-to centroid of one tooth ──
function flyToTooth(label){
  if(!mesh || !window._lastLabels) return;
  if(label === null){
    camera.position.copy(initialCamPos);
    controls.target.copy(initialTarget);
    controls.update();
    return;
  }
  const verts = mesh.geometry.attributes.position.array;
  const labels = window._lastLabels;
  let cx=0,cy=0,cz=0,n=0;
  let minX=Infinity,minY=Infinity,minZ=Infinity,maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
  for(let i=0;i<labels.length;i++){
    if(labels[i] !== label) continue;
    const x=verts[i*3], y=verts[i*3+1], z=verts[i*3+2];
    cx+=x; cy+=y; cz+=z; n++;
    if(x<minX)minX=x; if(x>maxX)maxX=x;
    if(y<minY)minY=y; if(y>maxY)maxY=y;
    if(z<minZ)minZ=z; if(z>maxZ)maxZ=z;
  }
  if(!n) return;
  cx/=n; cy/=n; cz/=n;
  const span = Math.max(maxX-minX, maxY-minY, maxZ-minZ);
  const target = new THREE.Vector3(cx, cy, cz);
  const dir    = camera.position.clone().sub(controls.target).normalize();
  camera.position.copy(target).addScaledVector(dir, Math.max(span * 4, 30));
  controls.target.copy(target);
  controls.update();
}

function selectTooth(label){
  // Click selected tooth again → clear
  if(selectedTooth === label) label = null;
  selectedTooth = label;
  recolor();
  flyToTooth(label);
  // Update legend highlighting
  document.querySelectorAll('.tooth').forEach(el => {
    el.classList.toggle('selected', label !== null && +el.dataset.label === label);
  });
}

// ── Viewer controls ──
$('btn-solid').onclick = () => { if(mesh) mesh.visible = true; if(wireframe) wireframe.visible = false; toggleActive('btn-solid'); };
$('btn-wire').onclick  = () => { if(mesh) mesh.visible = false; if(wireframe) wireframe.visible = true; toggleActive('btn-wire'); };
$('btn-hide-gum').onclick = function(){
  this.classList.toggle('active');
  gumHidden = this.classList.contains('active');
  recolor();
};
$('btn-reset-view').onclick = () => {
  selectedTooth = null;
  document.querySelectorAll('.tooth').forEach(el => el.classList.remove('selected'));
  recolor();
  flyToTooth(null);
};
function toggleActive(id){
  ['btn-solid','btn-wire'].forEach(b => $(b).classList.remove('active'));
  $(id).classList.add('active');
}

// ── Inference flow ──
$('submit').onclick = async () => {
  const file = $('file').files[0];
  if(!file){ setStatus('Pick a file first', 'error'); return; }
  const arch = $('arch').value;
  const token = $('token').value;

  $('submit').disabled = true;
  $('result').style.display = 'none';
  setStatus(`Uploading ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)…`);

  // Read file content client-side in parallel (for 3D view)
  const fileText = file.name.toLowerCase().endsWith('.obj') ? await file.text() : null;

  const fd = new FormData();
  fd.append('file', file);
  fd.append('arch', arch);

  const t0 = performance.now();
  try {
    const r = await fetch('/tooth-segmentation', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + token },
      body: fd,
    });
    if(!r.ok){
      setStatus(`Error ${r.status}: ${await r.text()}`, 'error');
      return;
    }
    const data = await r.json();
    const elapsed = ((performance.now() - t0)/1000).toFixed(1);
    setStatus(`Done in ${elapsed}s (server: ${data.inference_seconds}s)`, 'ok');
    renderResults(data, fileText);
  } catch(e) {
    setStatus('Network error: ' + e.message, 'error');
  } finally {
    $('submit').disabled = false;
  }
};

function renderResults(data, objText){
  $('result').style.display = 'block';

  $('meta').innerHTML = [
    ['Model',          data.model_used],
    ['Arch',           data.arch],
    ['Vertices',       data.vertex_count.toLocaleString()],
    ['Inference',      data.inference_seconds + 's'],
    ['Unique labels',  Object.keys(data.label_summary).length],
  ].map(([k,v])=>`<div class="meta-item"><div class="meta-label">${k}</div><div class="meta-value">${v}</div></div>`).join('');

  const entries = Object.entries(data.label_summary).sort((a,b)=>+a[0]-+b[0]);
  $('legend').innerHTML = entries.map(([id,count])=>{
    const c = FDI_COLORS[+id] || '#64748b';
    const name = (+id===0) ? 'gum' : `FDI ${id}`;
    return `<div class="tooth" data-label="${id}">
      <div class="swatch" style="background:${c}"></div>
      <span class="tooth-id">${name}</span>
      <span class="tooth-count">${count.toLocaleString()}</span>
    </div>`;
  }).join('');

  // Wire legend clicks to selectTooth()
  document.querySelectorAll('.tooth').forEach(el => {
    el.onclick = () => selectTooth(+el.dataset.label);
  });

  // Cache summary for hover tooltip
  window._lastSummary = {};
  for(const [k,v] of entries) window._lastSummary[+k] = v;

  const preview = {...data, labels: data.labels.slice(0,40).concat([`…+${data.labels.length-40} more`])};
  $('raw').textContent = JSON.stringify(preview, null, 2);

  // 3D view — only if we have OBJ text
  const overlay = $('viewer-overlay');
  if(objText){
    if(!scene) initViewer();
    const { verts, faces } = parseOBJ(objText);
    if(verts.length / 3 !== data.labels.length){
      overlay.textContent = `Vertex count mismatch (mesh=${verts.length/3} vs labels=${data.labels.length}) — viewer disabled`;
      return;
    }
    window._lastLabels = data.labels;
    selectedTooth = null;
    gumHidden = false;
    $('btn-hide-gum').classList.remove('active');
    renderMesh(verts, faces, data.labels);
    overlay.textContent = `${(verts.length/3).toLocaleString()} vertices · ${(faces.length/3).toLocaleString()} faces · Hover for FDI · Click legend to isolate`;
  } else {
    overlay.textContent = 'STL viewer not supported — upload .obj for 3D segmentation view';
  }
}

// Health check on load
fetch('/health').then(r=>r.json()).then(h=>{
  if(h.status !== 'ok') setStatus(`Server degraded — model not loaded (device=${h.device || 'none'})`, 'error');
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def ui_root():
    return _UI_HTML


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/tooth-segmentation", dependencies=[Depends(_verify_token)])
async def tooth_segmentation_endpoint(
    file: UploadFile = File(..., description="STL or OBJ intraoral scan mesh"),
    arch: str        = Form("upper", description="'upper' or 'lower'"),
):
    """
    Segment individual teeth from an intraoral scan mesh.

    - Input:  STL or OBJ file of a single dental arch (multipart/form-data)
    - Output: Per-vertex FDI tooth labels

    FDI label legend:
      0        = gingiva / unlabeled
      11–18    = upper right teeth (UR8..UR1)
      21–28    = upper left teeth  (UL1..UL8)
      31–38    = lower left teeth  (LL1..LL8)
      41–48    = lower right teeth (LR8..LR1)
    """
    if tooth_seg is None:
        raise HTTPException(
            status_code=503,
            detail="Tooth segmentation model not loaded. Check server logs for setup instructions."
        )

    allowed_extensions = {".stl", ".obj"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}"
        )

    if arch not in ("upper", "lower"):
        raise HTTPException(status_code=400, detail="arch must be 'upper' or 'lower'")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    job_id = str(uuid.uuid4())
    log.info(f"[{job_id}] tooth-segmentation start — arch={arch} file={file.filename} size={len(raw)}B")

    try:
        result = tooth_seg.run(raw, file.filename or "input.obj", arch=arch)
    except RuntimeError as e:
        log.error(f"[{job_id}] inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    log.info(f"[{job_id}] done in {result['inference_seconds']}s — {result['vertex_count']} vertices")

    return {
        "job_id": job_id,
        "arch":   arch,
        **result,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting tooth segmentation server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
