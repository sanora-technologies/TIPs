#!/usr/bin/env python3
"""
Cascade inference server — ToothFairy3 fold 0, Stage 1 (3d_lowres) + Stage 2
(3d_cascade_fullres), both at checkpoint_best.

Usage:
    python cascade_server.py [--port 7866] [--device cuda|cpu]

Endpoints:
    GET  /health           → { ready, stage1, stage2 }
    POST /infer            multipart file=<cbct.zip>  → { job_id }
    GET  /status/{job_id}  → { status, progress, message }
    GET  /result/{job_id}  → result.zip  (stls/*.stl + result.json)

Why this file exists
--------------------
Vanilla `nnUNetPredictor.predict_from_files_sequential` does the cascade
prev-stage one-hot expansion at *volume* level
(`data_iterators.py: data = np.vstack((data, seg_onehot))`). At Stage 2's
0.3 mm spacing on a typical CBCT (~71 M voxels), 77 fg channels × float32 =
~22 GB. That OOMs a 16 GB box instantly.

This server keeps the prev-stage as a single int channel and does the
one-hot expansion **per patch on GPU** inside a custom sliding-window
loop, modelled on
`nnUNetPredictor._internal_predict_sliding_window_return_logits`. Peak
RAM stays under ~6 GB for the largest production scans we see.

See HARDWARE.md (next to this file) for memory/VRAM budget and what
configuration changes are required to enable mirroring TTA.
"""

import gc
import io
import json
import logging
import os
import shutil
import struct
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Tuple

# ── nnUNet env must be set BEFORE any nnunetv2 import ───────────────────────
# The trained models live in the cbct_training project, not in this dir.
REPO_ROOT = Path(__file__).resolve().parent
CBCT_TRAIN_ROOT = Path("/home/oaiz/Documents/sanora/cbct_training")
NNUNET_DATA = CBCT_TRAIN_ROOT / "nnunet_data"
os.environ.setdefault("nnUNet_raw",          str(NNUNET_DATA / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(NNUNET_DATA / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results",      str(NNUNET_DATA / "nnUNet_results"))
# Keep dataloader workers off — sequential is the only safe path on 16 GB RAM.
os.environ.setdefault("nnUNet_n_proc_DA", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")
os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "131072")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import mcubes
try:
    import dicom2nifti  # optional — only needed if a DICOM zip is uploaded
except ImportError:
    dicom2nifti = None
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
from tqdm import tqdm

# Patch torch.load for nnUNet checkpoints (weights_only=False)
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.helpers import empty_cache
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

logging.basicConfig(level=logging.INFO, format="[cascade] %(message)s")
log = logging.getLogger("cascade")

# ── model folders ──────────────────────────────────────────────────────────
RESULTS_ROOT = NNUNET_DATA / "nnUNet_results" / "Dataset001_ToothFairy3"
MODEL_STAGE1 = RESULTS_ROOT / "nnUNetTrainerToothfairyCascade__nnUNetResEncUNetMPlans__3d_lowres"
MODEL_STAGE2 = RESULTS_ROOT / "nnUNetTrainerToothfairyCascade__nnUNetResEncUNetMPlans__3d_cascade_fullres"

# Pre-resample input CBCT to no finer than this. Only down-resamples — never
# blows up a coarse scan. Stage 2 model spacing is 0.3 mm, so anything ≥ 0.3 mm
# is fed unchanged; finer scans get bounded so the OUTPUT volume stays sane.
_INPUT_SPACING_CAP_MM = 0.3

# ── label map (kept identical to fold0_server.py) ───────────────────────────
LABELS: dict[int, tuple[str, str, str]] = {
    1:  ("Lower Jawbone",          "structures", "#c8a87a"),
    2:  ("Upper Jawbone",          "structures", "#d4b896"),
    3:  ("L Inferior Alveolar",    "structures", "#f4d03f"),
    4:  ("R Inferior Alveolar",    "structures", "#f4d03f"),
    5:  ("L Maxillary Sinus",      "structures", "#5dade2"),
    6:  ("R Maxillary Sinus",      "structures", "#5dade2"),
    7:  ("Pharynx",                "structures", "#a569bd"),
    8:  ("Bridge",                 "structures", "#b0c4d8"),
    9:  ("Crown",                  "structures", "#d0d8e0"),
    10: ("Implant",                "structures", "#8090a0"),
    11: ("#11 · UR Central Incisor",  "toothseg",   "#7edaf8"),
    12: ("#12 · UR Lateral Incisor",  "toothseg",   "#50c8f5"),
    13: ("#13 · UR Canine",           "toothseg",   "#20aaee"),
    14: ("#14 · UR 1st Premolar",     "toothseg",   "#1088e0"),
    15: ("#15 · UR 2nd Premolar",     "toothseg",   "#2068d8"),
    16: ("#16 · UR 1st Molar",        "toothseg",   "#3050cc"),
    17: ("#17 · UR 2nd Molar",        "toothseg",   "#4038c4"),
    18: ("#18 · UR Wisdom",           "toothseg",   "#5028bc"),
    19: ("#21 · UL Central Incisor",  "toothseg",   "#c0f060"),
    20: ("#22 · UL Lateral Incisor",  "toothseg",   "#98e048"),
    21: ("#23 · UL Canine",           "toothseg",   "#70d038"),
    22: ("#24 · UL 1st Premolar",     "toothseg",   "#48c028"),
    23: ("#25 · UL 2nd Premolar",     "toothseg",   "#28a820"),
    24: ("#26 · UL 1st Molar",        "toothseg",   "#189028"),
    25: ("#27 · UL 2nd Molar",        "toothseg",   "#107830"),
    26: ("#28 · UL Wisdom",           "toothseg",   "#186038"),
    27: ("#31 · LL Central Incisor",  "toothseg",   "#f8e840"),
    28: ("#32 · LL Lateral Incisor",  "toothseg",   "#f8c830"),
    29: ("#33 · LL Canine",           "toothseg",   "#f8a820"),
    30: ("#34 · LL 1st Premolar",     "toothseg",   "#f88818"),
    31: ("#35 · LL 2nd Premolar",     "toothseg",   "#f86010"),
    32: ("#36 · LL 1st Molar",        "toothseg",   "#e84808"),
    33: ("#37 · LL 2nd Molar",        "toothseg",   "#d83010"),
    34: ("#38 · LL Wisdom",           "toothseg",   "#c81818"),
    35: ("#41 · LR Central Incisor",  "toothseg",   "#f870c8"),
    36: ("#42 · LR Lateral Incisor",  "toothseg",   "#e850b8"),
    37: ("#43 · LR Canine",           "toothseg",   "#d830a8"),
    38: ("#44 · LR 1st Premolar",     "toothseg",   "#c81898"),
    39: ("#45 · LR 2nd Premolar",     "toothseg",   "#b80898"),
    40: ("#46 · LR 1st Molar",        "toothseg",   "#9810a8"),
    41: ("#47 · LR 2nd Molar",        "toothseg",   "#7818b8"),
    42: ("#48 · LR Wisdom",           "toothseg",   "#5828c8"),
    46: ("UR Cen Incisor Pulp",    "pulp",       "#e57373"),
    47: ("UR Lat Incisor Pulp",    "pulp",       "#e57373"),
    48: ("UR Canine Pulp",         "pulp",       "#e57373"),
    49: ("UR 1st Premolar Pulp",   "pulp",       "#e57373"),
    50: ("UR 2nd Premolar Pulp",   "pulp",       "#e57373"),
    51: ("UR 1st Molar Pulp",      "pulp",       "#e57373"),
    52: ("UR 2nd Molar Pulp",      "pulp",       "#e57373"),
    53: ("UR Wisdom Pulp",         "pulp",       "#e57373"),
    54: ("UL Cen Incisor Pulp",    "pulp",       "#e57373"),
    55: ("UL Lat Incisor Pulp",    "pulp",       "#e57373"),
    56: ("UL Canine Pulp",         "pulp",       "#e57373"),
    57: ("UL 1st Premolar Pulp",   "pulp",       "#e57373"),
    58: ("UL 2nd Premolar Pulp",   "pulp",       "#e57373"),
    59: ("UL 1st Molar Pulp",      "pulp",       "#e57373"),
    60: ("UL 2nd Molar Pulp",      "pulp",       "#e57373"),
    61: ("UL Wisdom Pulp",         "pulp",       "#e57373"),
    62: ("LL Cen Incisor Pulp",    "pulp",       "#e57373"),
    63: ("LL Lat Incisor Pulp",    "pulp",       "#e57373"),
    64: ("LL Canine Pulp",         "pulp",       "#e57373"),
    65: ("LL 1st Premolar Pulp",   "pulp",       "#e57373"),
    66: ("LL 2nd Molar Pulp",      "pulp",       "#e57373"),
    67: ("LL 1st Molar Pulp",      "pulp",       "#e57373"),
    68: ("LL 2nd Molar Pulp",      "pulp",       "#e57373"),
    69: ("LL Wisdom Pulp",         "pulp",       "#e57373"),
    70: ("LR Cen Incisor Pulp",    "pulp",       "#e57373"),
    71: ("LR Lat Incisor Pulp",    "pulp",       "#e57373"),
    72: ("LR Canine Pulp",         "pulp",       "#e57373"),
    73: ("LR 1st Premolar Pulp",   "pulp",       "#e57373"),
    74: ("LR 2nd Premolar Pulp",   "pulp",       "#e57373"),
    75: ("LR 1st Molar Pulp",      "pulp",       "#e57373"),
    76: ("LR 2nd Molar Pulp",      "pulp",       "#e57373"),
    77: ("LR Wisdom Pulp",         "pulp",       "#e57373"),
}

# ── GPU resampling patch (mirrors fold0_server.py for nnUNet's resample step) ─
def _apply_gpu_patch() -> None:
    try:
        from nnunetv2.preprocessing.resampling import default_resampling as _mod
        if getattr(_mod.resample_data_or_seg_to_shape, "__gpu_patched", False):
            return
        _orig = _mod.resample_data_or_seg_to_shape

        def _fast(data, new_shape, current_spacing, new_spacing,
                  is_seg=False, order=3, order_z=0, force_separate_z=None):
            new_shape = tuple(int(s) for s in new_shape)
            is_t  = isinstance(data, torch.Tensor)
            arr   = (data.cpu().numpy() if (is_t and data.device.type != "cpu")
                     else (data.numpy() if is_t else data))
            if arr.shape[1:] == new_shape:
                return data
            mode  = "nearest" if is_seg else "trilinear"
            extra = {} if is_seg else {"align_corners": False}
            try:
                with torch.no_grad():
                    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).cuda()
                    r = F.interpolate(t, size=new_shape, mode=mode, **extra)
                    out = r.squeeze(0).cpu().numpy()
                    del t, r; torch.cuda.empty_cache()
                return np.round(out).astype(arr.dtype) if is_seg else out
            except RuntimeError:
                torch.cuda.empty_cache()
                return _orig(data, new_shape, current_spacing, new_spacing,
                             is_seg, order, order_z, force_separate_z)

        _fast.__gpu_patched = True
        _mod.resample_data_or_seg_to_shape = _fast
        log.info("GPU resampling patch applied")
    except Exception as e:
        log.warning(f"GPU patch skipped: {e}")


# ── glibc memory release ────────────────────────────────────────────────────
_libc = None
def _malloc_trim() -> None:
    global _libc
    try:
        if _libc is None:
            import ctypes
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except Exception:
        pass


# ── input resampler ─────────────────────────────────────────────────────────
def _resample_input_nifti(src: Path) -> Path:
    img = sitk.ReadImage(str(src))
    orig_spacing = img.GetSpacing()
    if min(orig_spacing) >= _INPUT_SPACING_CAP_MM:
        return src
    orig_size = img.GetSize()
    new_size = [max(1, int(round(orig_size[i] * orig_spacing[i] / _INPUT_SPACING_CAP_MM)))
                for i in range(3)]
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([_INPUT_SPACING_CAP_MM] * 3)
    rs.SetSize(new_size)
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetOutputDirection(img.GetDirection())
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(-1000)
    out = rs.Execute(img)
    del img
    dst = src.parent / ("r_" + src.name)
    sitk.WriteImage(out, str(dst))
    del out
    log.info(f"Input resampled {orig_size}@{[f'{s:.2f}' for s in orig_spacing]}mm"
             f" → {new_size}@{_INPUT_SPACING_CAP_MM}mm")
    return dst


# ── predictor singletons ────────────────────────────────────────────────────
_pred_stage1: nnUNetPredictor | None = None
_pred_stage2: nnUNetPredictor | None = None
_pred_lock = threading.Lock()
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_predictor(model_folder: Path, use_mirroring: bool) -> nnUNetPredictor:
    p = nnUNetPredictor(
        tile_step_size=0.6,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=True,
        device=_DEVICE,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    p.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(0,),
        checkpoint_name="checkpoint_best.pth",
    )
    return p


def _get_predictors() -> Tuple[nnUNetPredictor, nnUNetPredictor]:
    global _pred_stage1, _pred_stage2
    if _pred_stage1 is not None and _pred_stage2 is not None:
        return _pred_stage1, _pred_stage2
    with _pred_lock:
        if _pred_stage1 is None or _pred_stage2 is None:
            _apply_gpu_patch()
            log.info(f"Loading Stage 1 (3d_lowres) on {_DEVICE} …")
            _pred_stage1 = _build_predictor(MODEL_STAGE1, use_mirroring=False)
            log.info(f"Loading Stage 2 (3d_cascade_fullres) on {_DEVICE} …")
            _pred_stage2 = _build_predictor(MODEL_STAGE2, use_mirroring=False)
            # Pre-move both networks to GPU so the first inference doesn't pay
            # a multi-second CPU→GPU weight transfer.
            if _DEVICE.type == "cuda":
                _pred_stage1.network = _pred_stage1.network.to(_DEVICE).eval()
                _pred_stage2.network = _pred_stage2.network.to(_DEVICE).eval()
            log.info("Predictors ready (networks on GPU).")
    return _pred_stage1, _pred_stage2


# ── fast Stage-1 predict (GPU argmax — no CPU logits detour) ────────────────
def _stage1_predict_fast(
    predictor: nnUNetPredictor,
    ct_input_file: Path,
    out_file: Path,
    progress_cb=None,
) -> None:
    """
    Replaces predict_from_files_sequential for Stage 1. Vanilla nnUNet moves
    the 78-channel float32 logits to CPU before argmax/resample — that's the
    slowness you see between "20/20 patches done" and "Stage 1 produced …".
    Here we argmax on GPU (only 1 channel of uint8 leaves the GPU), then
    resample the seg back.
    """
    plans_mgr = predictor.plans_manager
    cfg_mgr = predictor.configuration_manager
    ds_json = predictor.dataset_json
    pp = DefaultPreprocessor(verbose=False)

    if progress_cb:
        progress_cb("Stage 1 preprocess (resample CT to 0.45 mm)…")
    data_np, _, props = pp.run_case([str(ct_input_file)], None, plans_mgr, cfg_mgr, ds_json)
    data_t = torch.from_numpy(data_np).contiguous()
    del data_np
    log.info(f"Stage 1 preprocessed CT: {tuple(data_t.shape)}")

    if progress_cb:
        progress_cb("Stage 1 sliding window…")
    # predict_sliding_window_return_logits moves the network to device and
    # wraps the loop in autocast(fp16). Returns logits on GPU.
    logits = predictor.predict_sliding_window_return_logits(data_t)
    del data_t
    log.info(f"Stage 1 logits: shape={tuple(logits.shape)} device={logits.device}")

    if progress_cb:
        progress_cb("Stage 1 argmax on GPU…")
    if logits.is_cuda:
        seg_t = logits.argmax(0).to(torch.uint8)
        del logits
        torch.cuda.empty_cache()
        seg_np = seg_t.cpu().numpy()
        del seg_t
    else:
        seg_np = logits.argmax(0).to(torch.uint8).numpy()
        del logits
    gc.collect()

    if progress_cb:
        progress_cb("Stage 1 resample seg back to input spacing…")
    target_shape = props["shape_after_cropping_and_before_resampling"]
    if list(seg_np.shape) != list(target_shape):
        spacing_tp = [props["spacing"][i] for i in plans_mgr.transpose_forward]
        current_spacing = (
            cfg_mgr.spacing
            if len(cfg_mgr.spacing) == len(target_shape)
            else [spacing_tp[0], *cfg_mgr.spacing]
        )
        seg_resampled = cfg_mgr.resampling_fn_seg(
            seg_np[None], target_shape, current_spacing, spacing_tp
        )
        if isinstance(seg_resampled, torch.Tensor):
            seg_resampled = seg_resampled.cpu().numpy()
        seg_np = seg_resampled[0].astype(np.uint8)
        del seg_resampled

    seg_full = np.zeros(props["shape_before_cropping"], dtype=np.uint8)
    seg_full = insert_crop_into_image(seg_full, seg_np, props["bbox_used_for_cropping"])
    if isinstance(seg_full, torch.Tensor):
        seg_full = seg_full.cpu().numpy()
    seg_full = seg_full.transpose(plans_mgr.transpose_backward)
    del seg_np

    if progress_cb:
        progress_cb("Stage 1 write NIfTI…")
    rw = plans_mgr.image_reader_writer_class()
    rw.write_seg(seg_full, str(out_file), props)
    del seg_full
    gc.collect()
    _malloc_trim()


# ── custom Stage-2 sliding window with per-patch one-hot expansion ──────────
def _stage2_predict_logits(
    predictor: nnUNetPredictor,
    ct_4d: torch.Tensor,     # (1, X, Y, Z) float32 — preprocessed CT at model spacing
    seg_prev_4d: torch.Tensor,  # (1, X, Y, Z) int — Stage 1 seg at model spacing
) -> torch.Tensor:
    """
    Replaces predictor._internal_predict_sliding_window_return_logits for the
    cascade case. Holds the CT (1 ch float16) and Stage 1 seg (1 ch uint8) on
    device; one-hot expands the seg to 77 fg channels **per patch** so the
    78-channel input never materializes at volume scale.
    """
    device = predictor.device
    cfg_mgr = predictor.configuration_manager
    label_mgr = predictor.label_manager
    fg_labels = label_mgr.foreground_labels  # list[int], len = 77
    patch_size = tuple(cfg_mgr.patch_size)
    n_classes = label_mgr.num_segmentation_heads

    # Pad CT and seg identically so slicers line up.
    ct_padded, _ = pad_nd_image(ct_4d, patch_size, "constant", {"value": 0}, True, None)
    seg_padded, slicer_revert = pad_nd_image(seg_prev_4d, patch_size, "constant", {"value": 0}, True, None)

    slicers = predictor._internal_get_sliding_window_slicers(ct_padded.shape[1:])
    log.info(f"Stage 2 sliding window: {len(slicers)} patches, vol={tuple(ct_padded.shape)}")

    # Move the network to GPU. nnUNet's normal predict path does this inside
    # predict_sliding_window_return_logits — since we built our own sliding
    # window we have to do it ourselves, otherwise weight stays on CPU and
    # the conv reports a misleading "Half / float" error.
    predictor.network = predictor.network.to(device)
    net_param = next(predictor.network.parameters())
    net_dtype = net_param.dtype
    log.info(f"Stage 2 network: dtype={net_dtype} device={net_param.device}")
    ct_padded = ct_padded.to(device, dtype=net_dtype, non_blocking=True)
    seg_padded = seg_padded.to(device, dtype=torch.uint8, non_blocking=True)

    # Lookup table: label_id → channel index in the one-hot.
    # Built once on device for fast per-patch indexing.
    max_label = int(max(fg_labels)) + 1
    label_to_ch = torch.full((max_label,), -1, dtype=torch.long, device=device)
    for ci, lbl in enumerate(fg_labels):
        label_to_ch[int(lbl)] = ci
    n_fg = len(fg_labels)

    predicted_logits = torch.zeros(
        (n_classes, *ct_padded.shape[1:]), dtype=torch.half, device=device
    )
    n_predictions = torch.zeros(
        ct_padded.shape[1:], dtype=torch.half, device=device
    )
    gaussian = (
        compute_gaussian(patch_size, sigma_scale=1./8, value_scaling_factor=10, device=device)
        if predictor.use_gaussian else torch.tensor(1.0, device=device)
    )

    predictor.network.eval()
    # No autocast wrap — we explicitly pin workon to net_dtype above. autocast
    # under nested contexts was downcasting input to fp16 but leaving the bias
    # in fp32, causing the conv mismatch.
    with torch.inference_mode():
        for sl in tqdm(slicers, desc="stage2", disable=not predictor.allow_tqdm):
            # CT patch: (1, 1, Px, Py, Pz), already in net_dtype
            ct_patch = ct_padded[sl][None]
            # Seg patch as long indices: (1, 1, Px, Py, Pz)
            seg_patch = seg_padded[sl][None].to(torch.long)

            # Build one-hot for fg labels only, on GPU, matching network dtype.
            ch_idx = label_to_ch[seg_patch.clamp(0, max_label - 1)]  # (1, 1, Px, Py, Pz)
            valid = (ch_idx >= 0)
            onehot = torch.zeros(
                (1, n_fg, *seg_patch.shape[2:]),
                dtype=net_dtype, device=device,
            )
            ch_idx_safe = ch_idx.clamp(min=0)  # invalid → 0 (masked off via `valid`)
            onehot.scatter_(1, ch_idx_safe, valid.to(net_dtype))
            del ch_idx, ch_idx_safe, valid

            workon = torch.cat([ct_patch, onehot], dim=1)
            del onehot, seg_patch, ct_patch

            prediction = predictor._internal_maybe_mirror_and_predict(workon)[0]
            del workon

            if predictor.use_gaussian:
                prediction = prediction * gaussian
            # in-place += casts source to dest dtype (fp16 accumulator)
            predicted_logits[sl] += prediction
            n_predictions[sl[1:]] += gaussian
            del prediction

    torch.div(predicted_logits, n_predictions, out=predicted_logits)
    del n_predictions

    # Revert padding (keeps original pre-pad CT shape).
    # slicer_revert is (slice(None), *spatial_slices) — apply to all dims.
    predicted_logits = predicted_logits[
        (slice(None), *slicer_revert[1:])
    ].contiguous()
    return predicted_logits


def _stage2_export_to_nifti(
    predictor: nnUNetPredictor,
    logits: torch.Tensor,
    props: dict,
    out_file: Path,
) -> None:
    """Argmax on GPU, resample to original spacing, write NIfTI."""
    plans_mgr = predictor.plans_manager
    cfg_mgr = predictor.configuration_manager

    # Argmax on GPU (uint8 result, ~71x smaller than the float16 logits).
    seg_t = logits.argmax(0).to(torch.uint8)
    del logits
    torch.cuda.empty_cache()
    gc.collect()
    seg_np = seg_t.cpu().numpy()
    del seg_t

    # Resample from model spacing back to input pre-crop spacing.
    target_shape = props["shape_after_cropping_and_before_resampling"]
    if list(seg_np.shape) != list(target_shape):
        spacing_tp = [props["spacing"][i] for i in plans_mgr.transpose_forward]
        current_spacing = (
            cfg_mgr.spacing
            if len(cfg_mgr.spacing) == len(target_shape)
            else [spacing_tp[0], *cfg_mgr.spacing]
        )
        seg_4d = seg_np[None]
        seg_resampled = cfg_mgr.resampling_fn_seg(
            seg_4d, target_shape, current_spacing, spacing_tp
        )
        if isinstance(seg_resampled, torch.Tensor):
            seg_resampled = seg_resampled.cpu().numpy()
        seg_np = seg_resampled[0].astype(np.uint8)
        del seg_resampled

    seg_full = np.zeros(props["shape_before_cropping"], dtype=np.uint8)
    seg_full = insert_crop_into_image(seg_full, seg_np, props["bbox_used_for_cropping"])
    if isinstance(seg_full, torch.Tensor):
        seg_full = seg_full.cpu().numpy()
    seg_full = seg_full.transpose(plans_mgr.transpose_backward)
    del seg_np

    rw = plans_mgr.image_reader_writer_class()
    rw.write_seg(seg_full, str(out_file), props)
    del seg_full
    gc.collect()
    _malloc_trim()


# ── STL writer ──────────────────────────────────────────────────────────────
def _write_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    n = len(faces)
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", n))
        for tri in faces:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            nrm = np.cross(v1 - v0, v2 - v0)
            ln  = np.linalg.norm(nrm)
            if ln > 0:
                nrm /= ln
            f.write(struct.pack("<fff", *nrm.tolist()))
            f.write(struct.pack("<fff", *v0.tolist()))
            f.write(struct.pack("<fff", *v1.tolist()))
            f.write(struct.pack("<fff", *v2.tolist()))
            f.write(struct.pack("<H", 0))


# ── job store ───────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jlock = threading.Lock()


def _upd(job_id: str, **kw) -> None:
    with _jlock:
        _jobs[job_id].update(**kw)


# ── inference thread ────────────────────────────────────────────────────────
def _run_job(job_id: str, zip_bytes: bytes) -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="cascade_"))
    try:
        _upd(job_id, status="running", progress=2, message="Extracting zip…")

        # ── extract ──────────────────────────────────────────────────────
        zip_path = tmpdir / "input.zip"
        extracted = tmpdir / "extracted"
        zip_path.write_bytes(zip_bytes)
        extracted.mkdir()
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extracted)

        # ── locate / convert to NIfTI ───────────────────────────────────
        nii_in = tmpdir / "case_0000.nii.gz"
        nii_files = list(extracted.rglob("*.nii.gz")) + list(extracted.rglob("*.nii"))
        if nii_files:
            shutil.copy(nii_files[0], nii_in)
        else:
            _upd(job_id, progress=5, message="Converting DICOM → NIfTI…")
            dcm_dir: Path | None = None
            for cand in sorted([extracted] + list(extracted.rglob("*"))):
                if cand.is_dir() and any(cand.glob("*.dcm")):
                    dcm_dir = cand
                    break
            if dcm_dir is None:
                raise RuntimeError("No DICOM (.dcm) or NIfTI files found in zip")
            if dicom2nifti is None:
                raise RuntimeError(
                    "DICOM input received but dicom2nifti is not installed in this venv. "
                    "Either upload a .nii.gz zip, or `pip install dicom2nifti` in the active venv."
                )
            dicom2nifti.dicom_series_to_nifti(str(dcm_dir), str(nii_in), reorient_nifti=True)

        # ── pre-resample input (down-only) ──────────────────────────────
        _upd(job_id, progress=8, message="Resampling input to model spacing cap…")
        nii_in = _resample_input_nifti(nii_in)

        # nnUNet expects case_0000.nii.gz in an input folder
        nn_in = tmpdir / "nn_in"; nn_in.mkdir()
        shutil.copy(nii_in, nn_in / "case_0000.nii.gz")

        stage1_pred, stage2_pred = _get_predictors()
        amp = _DEVICE.type == "cuda"

        # ── STAGE 1: 3d_lowres with GPU-argmax export ───────────────────
        # Vanilla predict_from_files_sequential moves 78-ch fp32 logits to CPU
        # before argmax/resample — costs 50–120 s on this hardware. Our custom
        # path argmaxes on GPU and only moves 1 ch of uint8 to CPU (~30 MB).
        stage1_out = tmpdir / "stage1_out"; stage1_out.mkdir()
        stage1_seg = stage1_out / "case.nii.gz"  # nnUNet's case-id convention
        import time as _t
        _t1 = _t.time()
        _stage1_predict_fast(
            stage1_pred,
            nn_in / "case_0000.nii.gz",
            stage1_seg,
            progress_cb=lambda m: _upd(job_id, progress=15, message=m),
        )
        if amp:
            torch.cuda.empty_cache(); torch.cuda.synchronize()
        _malloc_trim()
        log.info(f"Stage 1 total: {_t.time() - _t1:.1f}s → {stage1_seg.name}")

        # ── STAGE 2: custom per-patch-onehot inference ──────────────────
        import time as _t
        _t0 = _t.time()
        _upd(job_id, progress=45,
             message="Stage 2 preprocess: resampling CT + Stage 1 seg to 0.3 mm…")
        log.info("Stage 2 preprocess starting (CT + Stage 1 seg → 0.3 mm). "
                 "This is the slow step on single-channel DDR5; please wait.")

        plans_mgr = stage2_pred.plans_manager
        cfg_mgr = stage2_pred.configuration_manager
        ds_json = stage2_pred.dataset_json
        pp = DefaultPreprocessor(verbose=False)

        # Preprocess CT *with* the Stage 1 seg so cropping/resampling stays aligned.
        data_np, seg_np, props = pp.run_case(
            [str(nn_in / "case_0000.nii.gz")],
            str(stage1_seg),
            plans_mgr, cfg_mgr, ds_json,
        )
        log.info(f"Stage 2 preprocess done in {_t.time() - _t0:.1f}s")
        # data_np: (1, X, Y, Z) float32 — preprocessed CT at Stage 2 model spacing
        # seg_np:  (1, X, Y, Z) int   — Stage 1 seg, resampled to model spacing
        log.info(f"Stage 2 preprocessed: CT={data_np.shape}, seg={seg_np.shape}")
        ct_t = torch.from_numpy(data_np).contiguous(); del data_np
        seg_t = torch.from_numpy(seg_np).contiguous(); del seg_np
        gc.collect()
        _malloc_trim()

        _upd(job_id, progress=50, message="Stage 2 sliding window (per-patch one-hot)…")
        # No autocast wrap here — _stage2_predict_logits pins the input dtype to
        # the network's parameter dtype, so we don't want a parent autocast
        # silently re-casting things.
        logits = _stage2_predict_logits(stage2_pred, ct_t, seg_t)
        del ct_t, seg_t
        gc.collect()
        if amp:
            torch.cuda.empty_cache()
        _malloc_trim()

        _upd(job_id, progress=68, message="Stage 2 export…")
        stage2_out = tmpdir / "stage2_out.nii.gz"
        _stage2_export_to_nifti(stage2_pred, logits, props, stage2_out)
        del logits
        if amp:
            torch.cuda.empty_cache(); torch.cuda.synchronize()
        gc.collect()
        _malloc_trim()

        # ── load final seg ───────────────────────────────────────────────
        _upd(job_id, progress=72, message="Loading segmentation…")
        seg_img = sitk.ReadImage(str(stage2_out))
        seg_arr_full = sitk.GetArrayFromImage(seg_img)
        sx, sy, sz = seg_img.GetSpacing()
        del seg_img

        MAX_DIM = 256
        max_dim = max(seg_arr_full.shape)
        if max_dim > MAX_DIM:
            step = int(np.ceil(max_dim / MAX_DIM))
            seg_arr = np.ascontiguousarray(seg_arr_full[::step, ::step, ::step])
            sx *= step; sy *= step; sz *= step
            log.info(f"Seg downsampled {seg_arr_full.shape} → {seg_arr.shape} (step={step})")
        else:
            seg_arr = np.ascontiguousarray(seg_arr_full)
        del seg_arr_full
        gc.collect()
        _malloc_trim()

        # ── marching cubes per label ─────────────────────────────────────
        _upd(job_id, progress=75, message="Generating 3D meshes…")
        stl_dir = tmpdir / "stls"; stl_dir.mkdir()
        result: dict = {
            "case":       "cbct_cascade",
            "toothseg":   {"base": "cbct_cascade", "layer": "toothseg",   "meshes": []},
            "pulp":       {"base": "cbct_cascade", "layer": "pulp",       "meshes": []},
            "structures": {"base": "cbct_cascade", "layer": "structures", "meshes": []},
            "perio":      {"base": "cbct_cascade", "teeth": [], "summary": {}},
        }
        n_labels = len(LABELS)
        for i, (lid, (name, layer, color)) in enumerate(LABELS.items()):
            bool_mask = (seg_arr == lid)
            if bool_mask.sum() < 100:
                del bool_mask
                continue
            mask = bool_mask.astype(np.float32)
            del bool_mask
            sigma = 0.3 if layer == "pulp" else 0.5
            mask = gaussian_filter(mask, sigma=sigma)
            try:
                verts, faces = mcubes.marching_cubes(mask, 0.5)
            except Exception as e:
                log.warning(f"mcubes label {lid}: {e}")
                del mask
                continue
            del mask
            if len(verts) == 0:
                continue
            verts[:, 0] *= sz
            verts[:, 1] *= sy
            verts[:, 2] *= sx
            _write_stl(stl_dir / f"{layer}_{lid}.stl", verts, faces)
            del verts, faces
            opacity = 0.75 if layer == "pulp" else (0.35 if lid in (5, 6, 7) else 1.0)
            result[layer]["meshes"].append({
                "label": lid, "name": name, "color": color, "opacity": opacity,
            })
            _upd(job_id, progress=75 + int(20 * (i + 1) / n_labels),
                 message=f"Meshed {layer} #{lid}…")
            if i % 10 == 0:
                gc.collect()
                _malloc_trim()

        del seg_arr
        gc.collect()
        _malloc_trim()

        # ── package zip ─────────────────────────────────────────────────
        _upd(job_id, progress=96, message="Packaging…")
        result_zip = tmpdir / "result.zip"
        with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("result.json", json.dumps(result))
            for stl in stl_dir.glob("*.stl"):
                z.write(stl, f"stls/{stl.name}")

        with _jlock:
            _jobs[job_id].update(
                status="done", progress=100, message="Done",
                _result=str(result_zip), _tmpdir=str(tmpdir))

    except Exception as e:
        log.exception(f"job {job_id} failed")
        _upd(job_id, status="error", message=str(e))
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── embedded HTML UI ───────────────────────────────────────────────────────
_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>CBCT Cascade Viewer</title>
<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>
<script src="https://unpkg.com/jszip@3.10.1/dist/jszip.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{display:flex;flex-direction:column;height:100vh;background:#111;color:#eee;
     font:13px/1.4 system-ui,sans-serif}
#hdr{padding:9px 16px;background:#1a1a1a;border-bottom:1px solid #333;
     display:flex;align-items:center;gap:12px;flex-shrink:0}
#hdr b{font-size:14px}#hdr span{font-size:11px;color:#777;flex:1}
#main{flex:1;display:flex;overflow:hidden}
#side{width:260px;background:#161616;border-right:1px solid #2a2a2a;padding:12px;
      display:flex;flex-direction:column;gap:8px;flex-shrink:0;overflow-y:auto}
#canvas-wrap{flex:1;position:relative}
#canvas-wrap canvas{display:block;width:100%!important;height:100%!important}
#drop{border:2px dashed #383838;border-radius:7px;padding:20px;text-align:center;
      cursor:pointer;transition:.2s;user-select:none}
#drop:hover,#drop.over{border-color:#555;background:#1d1d1d}
#drop.ok{border-color:#4caf50;background:#192519}
#drop .ic{font-size:24px;margin-bottom:5px}
#drop .lb{font-size:12px;color:#777}
#drop.ok .lb{color:#4caf50}
#run{padding:8px;background:#1565c0;border:none;border-radius:5px;color:#fff;
     font-size:13px;font-weight:600;cursor:pointer;width:100%}
#run:disabled{background:#2a2a2a;color:#555;cursor:not-allowed}
#prog{display:none;padding:2px 0}
#pbar-bg{height:5px;background:#252525;border-radius:3px;overflow:hidden;margin:5px 0 3px}
#pbar{height:100%;background:#1976d2;border-radius:3px;width:0;transition:width .4s}
#pmsg{font-size:10px;color:#777}
#err{display:none;font-size:11px;color:#ef5350;background:#1e1212;
     padding:7px;border-radius:4px;line-height:1.5}
.sec{font-size:10px;font-weight:700;color:#666;text-transform:uppercase;
     letter-spacing:.06em;margin-top:4px}
.grp{margin-bottom:2px}
.grp-hdr{display:flex;align-items:center;gap:6px;cursor:pointer;padding:3px 0;
         border-radius:3px;user-select:none}
.grp-hdr:hover{background:#1e1e1e}
.grp-hdr input{cursor:pointer;accent-color:#1976d2}
.grp-hdr .gtitle{font-size:12px;font-weight:600;flex:1}
.grp-hdr .gcnt{font-size:10px;color:#555}
.grp-hdr .arrow{font-size:9px;color:#555;transition:.15s}
.grp-hdr.open .arrow{transform:rotate(90deg)}
.grp-body{display:none;padding-left:18px}
.grp-body.open{display:block}
.mrow{display:flex;align-items:center;gap:5px;padding:2px 0;cursor:pointer}
.mrow:hover{background:#1a1a1a}
.mrow input{cursor:pointer;accent-color:#1976d2;flex-shrink:0}
.mrow .dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.mrow .mname{font-size:11px;color:#bbb;flex:1;white-space:nowrap;overflow:hidden;
              text-overflow:ellipsis}
#reset{padding:4px 10px;background:#252525;border:1px solid #383838;border-radius:4px;
       color:#ccc;cursor:pointer;font-size:11px}
#placeholder{display:flex;align-items:center;justify-content:center;
             height:100%;color:#383838;font-size:13px}
hr.div{border:none;border-top:1px solid #222;margin:4px 0}
</style>
</head>
<body>
<div id="hdr">
  <b>CBCT Cascade Viewer</b>
  <span>ToothFairy3 &middot; ResEncM &middot; 3d_lowres &rarr; 3d_cascade_fullres &middot; fold 0 &middot; checkpoint_best</span>
  <button id="reset" style="display:none" onclick="resetUI()">New scan</button>
</div>
<div id="main">
  <div id="side">
    <div id="drop" ondragover="ev(event,'over')" ondragleave="ev(event,'leave')"
         ondrop="ev(event,'drop')" onclick="document.getElementById('fi').click()">
      <div class="ic">📂</div>
      <div class="lb">Drop CBCT zip here<br>or click to browse</div>
    </div>
    <input id="fi" type="file" accept=".zip" style="display:none"
           onchange="pickFile(this.files[0])"/>
    <button id="run" disabled onclick="runInference()">Run Cascade Inference</button>
    <div style="display:flex;align-items:center;gap:6px;margin:2px 0">
      <hr style="flex:1;border:none;border-top:1px solid #2a2a2a"/>
      <span style="font-size:10px;color:#444">or</span>
      <hr style="flex:1;border:none;border-top:1px solid #2a2a2a"/>
    </div>
    <button id="load-btn" onclick="document.getElementById('rf').click()"
            style="padding:7px;background:#111e11;border:1px solid #2a472a;border-radius:5px;
                   color:#7bc87b;font-size:12px;font-weight:600;cursor:pointer;width:100%">
      Load Saved Result
    </button>
    <input id="rf" type="file" accept=".zip" style="display:none"
           onchange="loadSavedResult(this.files[0])"/>
    <div id="prog">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#666">
        <span id="plbl"></span><span id="ppct"></span></div>
      <div id="pbar-bg"><div id="pbar"></div></div>
      <div id="pmsg"></div>
    </div>
    <div id="err"></div>

    <div id="ctrl" style="display:none">
      <div class="sec">Visibility</div>
      <div class="grp" id="g-teeth">
        <div class="grp-hdr open" onclick="toggleGrp('teeth')">
          <input type="checkbox" id="ga-teeth" checked onchange="grpAll('teeth',this.checked)">
          <span class="gtitle">Teeth</span>
          <span class="gcnt" id="gn-teeth"></span>
          <span class="arrow">&#9654;</span>
        </div>
        <div class="grp-body open" id="gb-teeth"></div>
      </div>
      <div class="grp" id="g-pulp">
        <div class="grp-hdr" onclick="toggleGrp('pulp')">
          <input type="checkbox" id="ga-pulp" checked onchange="grpAll('pulp',this.checked)">
          <span class="gtitle">Pulp / Canals</span>
          <span class="gcnt" id="gn-pulp"></span>
          <span class="arrow">&#9654;</span>
        </div>
        <div class="grp-body" id="gb-pulp"></div>
      </div>
      <div class="grp" id="g-jaw">
        <div class="grp-hdr open" onclick="toggleGrp('jaw')">
          <input type="checkbox" id="ga-jaw" checked onchange="grpAll('jaw',this.checked)">
          <span class="gtitle">Jaw Bones</span>
          <span class="gcnt" id="gn-jaw"></span>
          <span class="arrow">&#9654;</span>
        </div>
        <div class="grp-body open" id="gb-jaw"></div>
      </div>
      <div class="grp" id="g-other">
        <div class="grp-hdr" onclick="toggleGrp('other')">
          <input type="checkbox" id="ga-other" checked onchange="grpAll('other',this.checked)">
          <span class="gtitle">Canals / Sinuses / Other</span>
          <span class="gcnt" id="gn-other"></span>
          <span class="arrow">&#9654;</span>
        </div>
        <div class="grp-body" id="gb-other"></div>
      </div>
      <hr class="div"/>
      <label class="mrow">
        <input type="checkbox" id="chk-rotate" checked onchange="autoRotate=this.checked">
        <span class="mname">Auto-rotate</span>
      </label>
    </div>
  </div>
  <div id="canvas-wrap">
    <div id="placeholder">Drop a CBCT zip and click Run Cascade Inference</div>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let selFile = null, pollTmr = null;
let renderer, scene, camera, controls;
let meshObjects = [];
window.autoRotate = true;

function labelGroup(layer, label) {
  if (layer === 'toothseg') return 'teeth';
  if (layer === 'pulp')     return 'pulp';
  if (label === 1 || label === 2) return 'jaw';
  return 'other';
}

window.ev = (e, t) => {
  e.preventDefault();
  const d = document.getElementById('drop');
  if (t === 'over')  d.classList.add('over');
  if (t === 'leave') d.classList.remove('over');
  if (t === 'drop')  { d.classList.remove('over'); const f=e.dataTransfer.files[0]; if(f) pickFile(f); }
};
window.pickFile = f => {
  selFile = f;
  const d = document.getElementById('drop');
  d.classList.add('ok');
  d.querySelector('.lb').textContent = f.name + ' (' + (f.size/1048576).toFixed(1) + ' MB)';
  document.getElementById('run').disabled = false;
};

function parseSTL(buf) {
  const v = new DataView(buf), n = v.getUint32(80, true);
  const verts = new Float32Array(n*9), idx = new Uint32Array(n*3);
  let o = 84;
  for (let t=0;t<n;t++) {
    o += 12;
    const b = t*9;
    for (let k=0;k<3;k++) {
      const vi = b+k*3;
      verts[vi]   = v.getFloat32(o,    true);
      verts[vi+1] = v.getFloat32(o+4,  true);
      verts[vi+2] = v.getFloat32(o+8,  true);
      o += 12;
    }
    o += 2;
    const fi=t*3; idx[fi]=fi; idx[fi+1]=fi+1; idx[fi+2]=fi+2;
  }
  return {verts,idx};
}

window.runInference = async function() {
  if (!selFile) return;
  document.getElementById('run').disabled = true;
  document.getElementById('err').style.display = 'none';
  setProg(true);
  try {
    upd('uploading', 2, 'Uploading…');
    const fd = new FormData(); fd.append('file', selFile);
    const r  = await fetch('/infer', {method:'POST', body:fd});
    if (!r.ok) throw new Error('Upload failed: ' + r.statusText);
    const {job_id} = await r.json();

    upd('running', 5, 'Cascade running…');
    await new Promise((res,rej) => {
      pollTmr = setInterval(async () => {
        try {
          const s = await (await fetch('/status/'+job_id)).json();
          upd('running', s.progress||0, s.message||'');
          if (s.status==='done')  {clearInterval(pollTmr); res();}
          if (s.status==='error') {clearInterval(pollTmr); rej(new Error(s.message));}
        } catch(e){clearInterval(pollTmr); rej(e);}
      }, 2500);
    });

    upd('parsing', 97, 'Parsing meshes…');
    const zr = await fetch('/result/'+job_id);
    if (!zr.ok) throw new Error('Download failed');
    const buf = await zr.arrayBuffer();
    const zip = await JSZip.loadAsync(buf);
    const meta = JSON.parse(await zip.file('result.json').async('string'));

    const all = [];
    for (const layer of ['toothseg','pulp','structures']) {
      for (const m of (meta[layer]?.meshes||[])) {
        const sf = zip.file('stls/'+layer+'_'+m.label+'.stl');
        if (!sf) continue;
        const ab = await sf.async('arraybuffer');
        try { all.push({...m, layer, geo:parseSTL(ab)}); } catch{}
      }
    }
    setProg(false);
    buildViewer(all);
  } catch(e) {
    setProg(false);
    const el=document.getElementById('err');
    el.textContent=e.message; el.style.display='block';
    document.getElementById('run').disabled=false;
  }
};

function upd(phase,pct,msg) {
  document.getElementById('plbl').textContent = phase;
  document.getElementById('ppct').textContent = pct+'%';
  document.getElementById('pbar').style.width = pct+'%';
  document.getElementById('pmsg').textContent = msg;
}
function setProg(on) {
  document.getElementById('prog').style.display = on ? 'block' : 'none';
}

function buildViewer(all) {
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('reset').style.display       = 'inline-block';
  document.getElementById('ctrl').style.display        = 'block';

  const wrap = document.getElementById('canvas-wrap');
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  renderer.setClearColor(0x0e0e0e);
  wrap.appendChild(renderer.domElement);

  scene  = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(40, wrap.clientWidth/wrap.clientHeight, 0.5, 50000);

  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const d1 = new THREE.DirectionalLight(0xffffff, 1.1); d1.position.set(200,400,200); scene.add(d1);
  const d2 = new THREE.DirectionalLight(0xffffff, 0.3); d2.position.set(-200,-200,-100); scene.add(d2);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06;

  let sx=0,sy=0,sz=0,cn=0;
  for (const m of all) {
    const v=m.geo.verts;
    for (let i=0;i<v.length;i+=3){sx+=v[i];sy+=v[i+1];sz+=v[i+2];cn++;}
  }
  const cx=sx/cn,cy=sy/cn,cz=sz/cn;
  controls.target.set(cx,cy,cz);
  camera.position.set(cx,cy,cz+320);

  const grpCounts = {teeth:0,pulp:0,jaw:0,other:0};
  meshObjects = [];

  for (const m of all) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(m.geo.verts,3));
    geo.setIndex(new THREE.BufferAttribute(m.geo.idx,1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(m.color||'#f5ede0'),
      transparent: m.opacity < 1,
      opacity: m.opacity,
      roughness: m.layer==='pulp' ? 0.55 : 0.22,
      metalness: 0.04,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
    const grp = labelGroup(m.layer, m.label);
    meshObjects.push({mesh, layer:m.layer, label:m.label, name:m.name, grp});
    grpCounts[grp]++;

    const row = document.createElement('label');
    row.className = 'mrow';
    row.title = m.name;
    row.innerHTML =
      `<input type="checkbox" checked data-grp="${grp}" data-lbl="${m.label}" data-layer="${m.layer}" onchange="rowVis(this)">` +
      `<span class="dot" style="background:${m.color||'#aaa'}"></span>` +
      `<span class="mname">${m.name}</span>`;
    document.getElementById('gb-'+grp).appendChild(row);
  }

  for (const [g,n] of Object.entries(grpCounts)) {
    document.getElementById('gn-'+g).textContent = n;
  }

  (function animate(){
    requestAnimationFrame(animate);
    if (window.autoRotate) camera.position.applyAxisAngle(new THREE.Vector3(0,1,0), 0.003);
    controls.update();
    renderer.render(scene,camera);
  })();

  window.addEventListener('resize', () => {
    camera.aspect = wrap.clientWidth/wrap.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  });
}

window.rowVis = function(cb) {
  const layer=cb.dataset.layer, lbl=+cb.dataset.lbl, vis=cb.checked;
  for (const o of meshObjects)
    if (o.layer===layer && o.label===lbl) { o.mesh.visible=vis; break; }
};

window.grpAll = function(grp, vis) {
  document.querySelectorAll(`[data-grp="${grp}"]`).forEach(cb => {
    cb.checked = vis;
    const layer=cb.dataset.layer, lbl=+cb.dataset.lbl;
    for (const o of meshObjects) if (o.layer===layer && o.label===lbl) {o.mesh.visible=vis;break;}
  });
};

window.toggleGrp = function(grp) {
  const hdr = document.querySelector(`#g-${grp} .grp-hdr`);
  const body = document.getElementById('gb-'+grp);
  hdr.classList.toggle('open');
  body.classList.toggle('open');
};

window.loadSavedResult = async function(f) {
  if (!f) return;
  document.getElementById('err').style.display = 'none';
  setProg(true);
  try {
    upd('loading', 20, 'Reading zip…');
    const buf = await f.arrayBuffer();
    upd('loading', 50, 'Parsing meshes…');
    const zip = await JSZip.loadAsync(buf);
    if (!zip.file('result.json'))
      throw new Error('Not a valid result zip — missing result.json');
    const meta = JSON.parse(await zip.file('result.json').async('string'));
    const all = [];
    for (const layer of ['toothseg','pulp','structures']) {
      for (const m of (meta[layer]?.meshes||[])) {
        const sf = zip.file('stls/'+layer+'_'+m.label+'.stl');
        if (!sf) continue;
        const ab = await sf.async('arraybuffer');
        try { all.push({...m, layer, geo:parseSTL(ab)}); } catch(e){}
      }
    }
    setProg(false);
    buildViewer(all);
  } catch(e) {
    setProg(false);
    const el = document.getElementById('err');
    el.textContent = e.message; el.style.display = 'block';
  }
};

window.resetUI = function() {
  if (pollTmr) clearInterval(pollTmr);
  if (renderer) {renderer.dispose(); renderer.domElement.remove(); renderer=null;}
  meshObjects = [];
  selFile = null;
  const d=document.getElementById('drop');
  d.classList.remove('ok');
  d.querySelector('.lb').textContent = 'Drop CBCT zip here\\nor click to browse';
  document.getElementById('run').disabled = true;
  document.getElementById('reset').style.display = 'none';
  document.getElementById('ctrl').style.display  = 'none';
  document.getElementById('placeholder').style.display = 'flex';
  document.getElementById('err').style.display   = 'none';
  document.getElementById('fi').value = '';
  document.getElementById('rf').value = '';
  setProg(false);
  for (const id of ['gb-teeth','gb-pulp','gb-jaw','gb-other'])
    document.getElementById(id).innerHTML = '';
};
</script>
</body>
</html>"""


# ── app ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="CBCT Cascade")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
def ui():
    return _UI


@app.get("/health")
def health():
    return {
        "ready": _pred_stage1 is not None and _pred_stage2 is not None,
        "stage1": MODEL_STAGE1.name,
        "stage2": MODEL_STAGE2.name,
    }


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    job_id = uuid.uuid4().hex[:10]
    with _jlock:
        _jobs[job_id] = {"status": "queued", "progress": 0, "message": "Queued"}
    threading.Thread(target=_run_job, args=(job_id, data), daemon=True).start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    with _jlock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {k: v for k, v in job.items() if not k.startswith("_")}


@app.get("/result/{job_id}")
def get_result(job_id: str):
    with _jlock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "done":
        raise HTTPException(400, f"not done (status={job['status']})")
    result_path = Path(job["_result"])
    td = job.get("_tmpdir")

    def iter_file():
        try:
            with open(result_path, "rb") as f:
                while chunk := f.read(65536):
                    yield chunk
        finally:
            if td:
                threading.Thread(target=shutil.rmtree, args=(td,),
                                 kwargs={"ignore_errors": True}, daemon=True).start()

    return StreamingResponse(
        iter_file(), media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="cbct_cascade_result.zip"',
            "Content-Length": str(result_path.stat().st_size),
        })


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",   type=int, default=7866)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    log.info(f"Pre-loading Stage 1 ({MODEL_STAGE1.name}) …")
    log.info(f"Pre-loading Stage 2 ({MODEL_STAGE2.name}) …")
    _get_predictors()
    log.info(f"Ready → http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
