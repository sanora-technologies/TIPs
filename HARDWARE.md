# Cascade Server — Hardware Budget & Tuning

Companion to [cascade_server.py](cascade_server.py). Explains the memory math, what the current 16 GB / 16 GB box can do, and what you'd need to flip TTA on.

---

## Current target box

| Component | Spec |
|---|---|
| CPU | Ryzen 5 7500F (6c/12t, AVX512) |
| GPU | RTX 5060 Ti, **16 GB VRAM** |
| RAM | **16 GB DDR5 4800 MT/s, single stick (1×16)** |
| Storage | 512 GB NVMe (DRAM-less, MAP1202) |

Two real bottlenecks here:

1. **Single-stick DDR5** runs single-channel → ~38 GB/s instead of ~75 GB/s. nnUNet preprocessing (resampling, normalization) is memory-bandwidth-bound. This is the *invisible* reason CPU/RAM gets pegged while the GPU coasts.
2. **16 GB RAM is the absolute minimum** for cascade inference. A vanilla `predict_from_files_sequential` call on Stage 2 needs **~22 GB** for the volume-level 77-channel one-hot alone (see "Why cascade OOMs out-of-the-box" below).

`cascade_server.py` is designed around both constraints.

---

## Memory budget — what's actually held in RAM/VRAM

For a representative production CBCT (16×12×10 cm FOV, native ~0.2 mm spacing, resampled to 0.3 mm cap → 533×400×333 ≈ 71 M voxels):

### RAM (CPU) at peak Stage 2 sliding window

| Item | Size | Notes |
|---|---|---|
| Input CBCT NIfTI (uint16/int16) | ~140 MB | Resampled SimpleITK image |
| Preprocessed CT (float32, 1 ch, padded) | ~285 MB | After DefaultPreprocessor |
| Stage 1 seg, model-spacing (uint8, 1 ch) | ~71 MB | Resampled in preprocessor |
| Inference framework overhead | ~1.5–2 GB | PyTorch, nnUNet, models loaded in RAM |
| OS + Python + libs | ~1.5 GB | |
| **Total peak RAM** | **~4–5 GB** | Comfortable on 16 GB |

### VRAM (GPU) at peak Stage 2 sliding window

| Item | Size | Notes |
|---|---|---|
| Model weights (ResEnc-M cascade) | ~120 MB | float16 |
| CT on device (float16, 1 ch) | ~140 MB | |
| Stage 1 seg on device (uint8) | ~71 MB | |
| `predicted_logits` (float16, 78 ch) | ~11 GB | Pre-allocated full-volume accumulator |
| `n_predictions` (float16, 1 ch) | ~140 MB | |
| Per-patch one-hot (float16, 77 ch, 64×160×160) | ~250 MB | Allocated/freed each patch |
| Patch forward activations (autocast fp16) | ~1.5–2 GB | |
| **Total peak VRAM** | **~13–14 GB** | Fits in 16 GB with safety margin |

The big VRAM line is `predicted_logits` (78 ch × Z×Y×X × 2 B). If you ever see CUDA OOM, the fix is to lower `_INPUT_SPACING_CAP_MM` in [cascade_server.py:107](cascade_server.py#L107) — bumping it from 0.3 to 0.4 cuts that 11 GB to ~4.6 GB.

### Why cascade OOMs out-of-the-box

`nnunetv2.inference.data_iterators.PreprocessAdapter.generate_train_batch()` does this at volume level:

```python
seg_onehot = convert_labelmap_to_one_hot(seg[0], foreground_labels, data.dtype)  # float32
data = np.vstack((data, seg_onehot))  # (1 CT + 77 fg) × float32 × 71 M voxels = 22 GB
```

This server bypasses that path entirely. Stage 2 uses [cascade_server.py `_stage2_predict_logits`](cascade_server.py) which keeps the prev-stage seg as a single uint8 channel on the GPU and one-hot expands it **per patch** (a 250 MB allocation that's reused every patch) before concatenating with the CT patch and running the network. No volume-level 78-channel tensor is ever materialized.

---

## TTA (test-time mirroring) — when to turn it on

MODEL.md mentions "full mirroring TTA (8 passes) for maximum accuracy." It's **off by default** in this server. Here's why and what flipping it on costs:

| Mode | Stage 1 latency | Stage 2 latency | Stage 2 peak VRAM |
|---|---|---|---|
| **No TTA** (current default) | ~2 min | ~6–10 min | ~14 GB |
| Stage 1 TTA only | ~10 min | ~6–10 min | ~14 GB |
| Stage 1 + Stage 2 TTA | ~10 min | **~25–40 min** | **~16–18 GB (OOM risk)** |

The Stage 2 problem is the 78-channel input. Mirroring averages 8 forward passes; for the cascade model that's much heavier in absolute terms than for Stage 1's 1-channel input. Patch activations are 8× larger in the worst case (per-axis cumulative averaging buffer), pushing us over the 16 GB VRAM limit on big patches.

### To enable TTA safely you need

| Want… | Need… |
|---|---|
| Stage 1 TTA only (~+0.5–1 Dice point on rare classes) | Current box can already do this. Set `use_mirroring=True` only in the Stage 1 builder in [cascade_server.py `_get_predictors`](cascade_server.py). Adds ~8 min per scan. |
| Stage 1 + Stage 2 TTA (full MODEL.md recipe) | **≥ 24 GB VRAM** (RTX 4090 / RTX 5090 / A5000). **≥ 32 GB RAM** strongly recommended (logits accumulator + temporary buffers). Set `use_mirroring=True` in both builders. |
| Stage 1 + Stage 2 TTA + sub-0.3 mm input fidelity | **≥ 32 GB VRAM** (A6000 / L40 / H100). **≥ 64 GB RAM**. Then lower `_INPUT_SPACING_CAP_MM` to e.g. 0.2 mm to preserve native CBCT detail end-to-end. |

---

## Recommended upgrade path (this exact box)

If you want a meaningfully faster / more accurate cascade without buying a new GPU:

1. **Add a second 16 GB DDR5 stick.** Single biggest win. Dual-channel doubles bandwidth → preprocessing and resampling drop ~40 % in wall-clock, and gives headroom for `nnUNet_n_proc_DA=2` background augmentation workers during inference (currently 0).
2. **Set `_INPUT_SPACING_CAP_MM` based on your scanner.** If your CBCTs ship at 0.3 mm natively, keep it at 0.3. If they ship at 0.2 mm or finer, leaving the cap at 0.3 already saves you. Going below 0.3 buys nothing — the Stage 2 model was trained at 0.3 mm.
3. **NVMe with DRAM.** The MAP1202 is DRAM-less; nnUNet's preprocessor reads/writes intermediate `.nii.gz` files multiple times per scan. A DRAM-equipped NVMe (SK Hynix P41, Samsung 990 Pro) drops I/O latency ~3×. Not critical, but visible on cold scans.
4. **GPU upgrade only if TTA is non-negotiable.** RTX 5060 Ti 16 GB is sufficient for production no-TTA cascade. Step up to 24 GB only if Dice deltas from Stage 2 TTA matter clinically.

---

## What the server does to stay inside the budget

Quick reference to the actual tricks in [cascade_server.py](cascade_server.py):

| Trick | Where | Saves |
|---|---|---|
| Per-patch one-hot on GPU (not volume-level on CPU) | `_stage2_predict_logits` | **~20 GB RAM** at Stage 2 |
| `predict_from_files_sequential` for Stage 1 (no worker procs) | `_run_job` Stage 1 path | ~2–3 GB RAM (no preprocessing fork) |
| `tile_step_size=0.6` (not 0.5) | `_build_predictor` | ~1.4× fewer tiles → ~30 % faster |
| `use_mirroring=False` | `_build_predictor` | 2× VRAM, 8× speed for both stages |
| Argmax on GPU before CPU transfer | `_stage2_export_to_nifti` | Avoids 22 GB float32 logits → CPU OOM |
| Pre-resample input to 0.3 mm cap (down-only) | `_resample_input_nifti` | Bounds output volume, no accuracy loss vs. model |
| Stride-downsample seg to MAX_DIM=256 before marching cubes | `_run_job` mesh stage | ~5× fewer mcubes calls, ~10 GB float saved |
| `malloc_trim(0)` after every big stage | `_malloc_trim` | Returns freed heap to OS (single-stick RAM is precious) |
| `expandable_segments:True` for PyTorch allocator | env at top of file | Prevents VRAM fragmentation across Stage 1 → Stage 2 |

---

## Running it

```bash
# Use the main TIPs venv (must have nnunetv2 + mcubes + dicom2nifti + fastapi)
source /home/oaiz/Documents/Sanora/dianexea_stack/TIPs/venv/bin/activate

# Make sure CUDA is visible
nvidia-smi

# Start
python /home/oaiz/Documents/Sanora/dianexea_stack/TIPs/cascade_server.py --port 7866

# Open http://localhost:7866 → drop zip → wait ~8–12 min → STL viewer
```

Expected end-to-end latency on the current box, no TTA, typical CBCT:
- Upload + unzip + resample: ~30 s
- Stage 1 inference: ~2 min
- Stage 2 preprocessing: ~30 s
- Stage 2 sliding window: ~6–8 min
- Stage 2 export + meshing: ~1 min
- **Total: ~10–12 min/scan**
