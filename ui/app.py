"""
Dental 3D viewer backend.
Run from project root: python ui/app.py
Open: http://localhost:7860
"""
import os
import struct
import zlib
from pathlib import Path

import numpy as np
import nibabel as nib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter, binary_closing, binary_dilation
from scipy.spatial import cKDTree
from numpy.linalg import eigh

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent          # project root (TIPs/)
UI_STATIC = Path(__file__).parent / "static"

# Layer name → filename inside output/{scan}/
LAYER_FILES = {
    "toothseg":   "toothseg.nii.gz",
    "pulp":       "pulp.nii.gz",
    "structures": "structures.nii.gz",
    "segmentator": "segmentation.nii.gz",
}

app = FastAPI()

# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_nii(base: str, layer: str) -> Path:
    """Return the NIfTI path for a (base, layer) pair, or None."""
    filename = LAYER_FILES.get(layer)
    if filename:
        p = ROOT / "output" / base / filename
        if p.exists():
            return p
    return None


def find_scans() -> list[dict]:
    """Return list of scan sets available in output/."""
    output_dir = ROOT / "output"
    if not output_dir.is_dir():
        return []

    scans_map: dict[str, list] = {}
    for d in output_dir.iterdir():
        if not d.is_dir():
            continue
        layers = []
        for key, filename in LAYER_FILES.items():
            if (d / filename).exists():
                layers.append(key)
        if layers:
            scans_map[d.name] = sorted(layers)

    # Sort newest first (most-recently modified folder at top)
    return [{"base": b, "layers": layers}
            for b, layers in sorted(
                scans_map.items(),
                key=lambda kv: (output_dir / kv[0]).stat().st_mtime,
                reverse=True,
            )]


def _taubin_smooth(vertices: np.ndarray, faces: np.ndarray,
                   iterations: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Taubin smoothing: shrink (λ>0) then inflate (μ<0) each iteration.
    Preserves volume while removing voxel staircase artefacts.
    Pure numpy — no external dependencies.
    """
    lam =  0.50
    mu  = -0.53

    v = vertices.copy()
    n_verts = len(v)

    # Directed half-edge pairs for every triangle edge (both directions)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2],
                           faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0],
                           faces[:, 0], faces[:, 1], faces[:, 2]])

    # Neighbour counts are topology-fixed — compute once outside the loop
    neighbour_cnt = np.zeros(n_verts, dtype=np.float32)
    np.add.at(neighbour_cnt, rows, 1.0)
    inv_cnt = (1.0 / np.maximum(neighbour_cnt, 1))[:, None]  # (V,1)

    neighbour_sum = np.zeros_like(v)

    for _ in range(iterations):
        # Shrink step (λ > 0): pull each vertex toward its neighbours
        neighbour_sum[:] = 0.0
        np.add.at(neighbour_sum, rows, v[cols])
        v += lam * (neighbour_sum * inv_cnt - v)

        # Inflate step (μ < 0): push back slightly to restore volume
        neighbour_sum[:] = 0.0
        np.add.at(neighbour_sum, rows, v[cols])
        v += mu * (neighbour_sum * inv_cnt - v)

    return v, faces


def _teeth_world_bbox(base: str):
    """Return (min_xyz, max_xyz) world-space bounding box of the toothseg layer, or None."""
    p = _resolve_nii(base, 'toothseg')
    if p is None:
        return None
    img  = nib.load(str(p))
    data = np.asarray(img.dataobj)
    vx   = np.argwhere(data > 0).astype(np.float32)
    if len(vx) == 0:
        return None
    ones  = np.ones((len(vx), 1), dtype=np.float32)
    world = (img.affine @ np.hstack([vx, ones]).T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def load_mesh(base: str, layer: str) -> dict:
    """Run marching cubes + Taubin smoothing on a segmentation, return vertices+faces."""
    nii_path = _resolve_nii(base, layer)
    if nii_path is None:
        raise HTTPException(404, f"No NIfTI found for base={base!r}, layer={layer!r}")

    img  = nib.load(str(nii_path))
    data = np.asarray(img.dataobj)

    # ── For bone layers, build a crop box around the dental arch ─────────────
    # Maxilla (label 1) covers the entire skull; mandible (label 2) extends to
    # the chin.  Keep only bone within a generous margin around the teeth bbox
    # so sinuses / skull cap / chin are clipped out.
    # Use ASYMMETRIC margins:
    #   +60 mm below  (inferior) — keeps full mandible body below root apices
    #   +25 mm above  (superior) — clips skull base while keeping palate
    #   +30 mm around (lateral + AP) — keeps full arch width
    bone_crop_bbox = None
    if layer in ('structures', 'segmentator'):
        bbox = _teeth_world_bbox(base)
        if bbox is not None:
            mn_xyz, mx_xyz = bbox[0].copy(), bbox[1].copy()
            # Determine which axis is most "vertical" from the affine
            # (the column of the affine whose absolute value is largest in Z)
            aff3 = np.abs(img.affine[:3, :3])
            vert_axis = int(np.argmax(aff3[2]))  # voxel axis most aligned with world-Z
            MARGIN_LAT   = 30.0
            MARGIN_BELOW = 60.0
            MARGIN_ABOVE = 25.0
            lo = mn_xyz - MARGIN_LAT
            hi = mx_xyz + MARGIN_LAT
            # Apply asymmetric superior/inferior margins on the vertical axis
            lo[vert_axis] = mn_xyz[vert_axis] - MARGIN_BELOW
            hi[vert_axis] = mx_xyz[vert_axis] + MARGIN_ABOVE
            bone_crop_bbox = (lo, hi)

    labels = np.unique(data)
    labels = labels[labels > 0]
    if len(labels) == 0:
        raise HTTPException(422, "Segmentation is empty (all zeros).")

    affine = img.affine

    # Sigma: 1.5 voxels gives smooth surfaces without over-blurring fine anatomy.
    # Structures (bone) get slightly more smoothing than teeth/pulp.
    sigma_map = {
        "structures":  3.0,   # bones: heavy smooth for clean shell appearance
        "segmentator": 2.5,
        "toothseg":    1.5,   # teeth: smooth but keep crown geometry
        "pulp":        1.5,
    }
    sigma = sigma_map.get(layer, 1.5)

    # Taubin iterations: more for bones (large flat surfaces), fewer for teeth
    taubin_map = {
        "structures":  25,
        "segmentator": 20,
        "toothseg":    12,
        "pulp":        10,
    }
    taubin_iters = taubin_map.get(layer, 12)

    meshes = []
    for lbl in labels:
        vol = (data == lbl).astype(np.uint8)
        if vol.sum() < 50:
            continue

        # ── Dilate bone masks to fill patchy tooth-root junction gaps ─────────
        if layer in ('structures', 'segmentator') and int(lbl) in (1, 2):
            vol = binary_closing(vol, iterations=8).astype(np.uint8)

        # ── Pre-smooth: Gaussian blur → smooth gradient field ────────────────
        scalar_field = gaussian_filter(vol.astype(np.float32), sigma=sigma)

        # ── Boundary mask for toothseg: zero out neighbour-label voxels ──────
        # Gaussian blur spreads ~3-4 voxels; if two teeth are that close their
        # blurred fields overlap and the 0.5 isosurface bridges them into one
        # mesh.  Setting neighbour voxels to 0 forces the isosurface to stay
        # strictly within each tooth's own space — same result as Diagnocat's
        # label-boundary-aware extraction, without the SDF roughness.
        if layer == 'toothseg':
            scalar_field[(data > 0) & (data != lbl)] = 0.0

        level = 0.5
        vmin, vmax = float(scalar_field.min()), float(scalar_field.max())
        if vmax - vmin < 1e-6 or vmax < level:
            continue
        verts, faces, _, _ = marching_cubes(scalar_field, level=level)

        # ── Transform voxel indices → world-space mm ─────────────────────────
        ones  = np.ones((len(verts), 1))
        world = (affine @ np.hstack([verts, ones]).T).T[:, :3].astype(np.float32)

        # ── Post-smooth: Taubin smoothing removes residual voxel artefacts ───
        world, faces = _taubin_smooth(world, faces, iterations=taubin_iters)

        # ── Crop bone mesh to dental-arch bounding box (removes skull/sinus) ─
        if bone_crop_bbox is not None and int(lbl) in (1, 2):
            mn, mx = bone_crop_bbox
            keep_v = np.all((world >= mn) & (world <= mx), axis=1)
            if keep_v.sum() < 10:
                continue   # nothing left after crop
            # Remap indices: only keep faces where ALL 3 vertices survive
            new_idx = np.full(len(world), -1, dtype=np.int32)
            new_idx[keep_v] = np.arange(keep_v.sum(), dtype=np.int32)
            keep_f = keep_v[faces[:, 0]] & keep_v[faces[:, 1]] & keep_v[faces[:, 2]]
            faces  = new_idx[faces[keep_f]]
            world  = world[keep_v]
            if len(faces) == 0:
                continue

        meshes.append({
            "label":    int(lbl),
            "vertices": world.flatten().tolist(),
            "faces":    faces.flatten().tolist(),
        })

    return {"base": base, "layer": layer, "meshes": meshes}


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/api/scans")
def api_scans():
    return find_scans()


@app.get("/api/mesh/{base}/{layer}")
def api_mesh(base: str, layer: str):
    return JSONResponse(load_mesh(base, layer))


@app.get("/api/crown/{base}/{label}")
def api_crown(
    base: str,
    label: int,
    offset_mm: float = Query(default=1.0, ge=0.3, le=4.0),
    sigma:     float = Query(default=0.5, ge=0.1, le=2.0),
    taubin:    int   = Query(default=5,   ge=0,   le=20),
):
    """
    Generate a crown/cap mesh for a single tooth label.
    The crown is the dilated outer shell of the tooth — ready to 3D-print
    as a dental cap.  Uses low sigma + few Taubin iterations to preserve
    occlusal surface detail (cusps, fissures).
    """
    nii_path = _resolve_nii(base, 'toothseg')
    if nii_path is None:
        raise HTTPException(404, "toothseg layer not found for this scan")

    img    = nib.load(str(nii_path))
    data   = np.asarray(img.dataobj)
    affine = img.affine

    tooth_mask = (data == label)
    if tooth_mask.sum() < 10:
        raise HTTPException(404, f"Label {label} not found in toothseg")

    # Voxel size (mm) — use mean of diagonal magnitudes for near-isotropic scans
    vox_mm = float(np.abs(np.diag(affine[:3, :3])).mean())
    dil_iters = max(1, round(offset_mm / vox_mm))

    # Dilate → outer crown volume; keep detail by using low Gaussian sigma
    outer = binary_dilation(tooth_mask, iterations=dil_iters).astype(np.uint8)
    scalar = gaussian_filter(outer.astype(np.float32), sigma=sigma)

    if float(scalar.max()) < 0.5:
        raise HTTPException(422, "Crown generation produced an empty mesh")

    verts, faces, _, _ = marching_cubes(scalar, level=0.5)

    ones  = np.ones((len(verts), 1))
    world = (affine @ np.hstack([verts, ones]).T).T[:, :3].astype(np.float32)
    world, faces = _taubin_smooth(world, faces, iterations=taubin)

    return JSONResponse({
        "label":    label,
        "vertices": world.flatten().tolist(),
        "faces":    faces.flatten().tolist(),
        "offset_mm": offset_mm,
        "vox_mm":   round(vox_mm, 3),
    })


@app.get("/api/perio/{base}")
def api_perio(base: str):
    return JSONResponse(_compute_perio(base))


def _compute_perio(base: str) -> dict:
    """
    For each segmented tooth, find the alveolar bone crest and compute:
      - bone_point      : world-space XYZ of the bone-crest centroid
      - long_axis       : unit vector from root toward crown (+ = crown)
      - ring_radius_mm  : actual tooth half-width at bone crest level
      - root_exposed_mm : estimated exposed root (bone loss) in mm
      - severity        : 'normal' | 'mild' | 'moderate' | 'severe'

    Key improvements over naïve global KD-tree:
      1. Per-tooth bone filtering:  only alveolar bone near the ROOT zone
         of each tooth is considered.  This prevents the hard palate / sinus
         floor from being picked as "bone" for maxillary teeth.
      2. Bone crest refined via a thin ±2 mm slice around the crest level,
         giving an accurate centroid on the tooth neck — not a smeared mean
         over the whole root.
      3. Ring radius from the slice cross-section (percentile-65, no inflation).
    """
    toothseg_path = _resolve_nii(base, 'toothseg')
    struct_path   = _resolve_nii(base, 'structures')
    if toothseg_path is None:
        raise HTTPException(404, "toothseg layer not found — run inference first")
    if struct_path is None:
        raise HTTPException(404, "structures layer not found — run inference first")

    tooth_img  = nib.load(str(toothseg_path))
    struct_img = nib.load(str(struct_path))
    tooth_data  = np.asarray(tooth_img.dataobj).astype(np.int32)
    struct_data = np.asarray(struct_img.dataobj).astype(np.int32)
    tooth_affine  = tooth_img.affine
    struct_affine = struct_img.affine

    # ── Build ALL bone voxels once (sub-sampled ×3 for speed) ────────────────
    bone_mask = (struct_data == 1) | (struct_data == 2)
    bone_vx   = np.argwhere(bone_mask[::3, ::3, ::3]).astype(np.float32) * 3
    if len(bone_vx) == 0:
        raise HTTPException(422, "No bone segmentation found in structures layer")
    ones_b          = np.ones((len(bone_vx), 1), dtype=np.float32)
    bone_world_all  = (struct_affine @ np.hstack([bone_vx, ones_b]).T).T[:, :3].astype(np.float64)

    FDI_TO_JAW = {
        **{i: 'maxilla'  for i in list(range(11, 19)) + list(range(21, 29))},
        **{i: 'mandible' for i in list(range(31, 39)) + list(range(41, 49))},
    }

    results = []
    for lbl in sorted(np.unique(tooth_data)):
        if lbl == 0:
            continue
        tooth_mask_arr = (tooth_data == lbl)
        if tooth_mask_arr.sum() < 30:
            continue

        # ── World coords for this tooth ───────────────────────────────────────
        tooth_vx    = np.argwhere(tooth_mask_arr).astype(np.float32)
        ones_t      = np.ones((len(tooth_vx), 1), dtype=np.float32)
        tooth_world = (tooth_affine @ np.hstack([tooth_vx, ones_t]).T).T[:, :3].astype(np.float64)

        # ── PCA → long axis (arbitrary direction at this point) ───────────────
        centroid  = tooth_world.mean(axis=0)
        centered  = tooth_world - centroid
        cov       = (centered.T @ centered) / max(len(centered) - 1, 1)
        _, eigvecs = eigh(cov)
        long_axis  = eigvecs[:, -1].astype(np.float64)

        proj              = centered @ long_axis
        tooth_length_mm   = float(proj.max() - proj.min())
        root_proj_val     = float(proj.min())

        # ── Orient long_axis: crown at + end ─────────────────────────────────
        # Use a rough nearby-bone centroid; bone should be on the root side.
        rough_near        = np.linalg.norm(bone_world_all - centroid, axis=1) < 20.0
        if rough_near.sum() < 10:
            continue
        rough_bone_center = bone_world_all[rough_near].mean(axis=0)
        if float((rough_bone_center - centroid) @ long_axis) > 0:
            long_axis = -long_axis
            proj      = -proj
            root_proj_val = float(proj.min())

        # ── Filter bone to ALVEOLAR region of THIS tooth ──────────────────────
        #
        #  Problem: structures label-1 (Maxilla) includes hard palate, sinus
        #  walls, etc.  For upper teeth the palate can be CLOSER to the root
        #  apices than the alveolar crest → global KD-tree picks the wrong bone.
        #
        #  Fix: keep only bone voxels that satisfy BOTH:
        #    (a) Within 14 mm perpendicular distance from the tooth's long axis
        #        (cuts out palate which is off-axis for incisors/premolars)
        #    (b) Projection along long_axis  <  root_proj + 65 % of tooth length
        #        (crown zone is excluded — alveolar bone cannot be above the CEJ)
        #
        b_cent      = bone_world_all - centroid          # (N, 3)
        b_proj      = b_cent @ long_axis                 # (N,)
        b_ax_comp   = np.outer(b_proj, long_axis)        # component along axis
        b_perp_dist = np.linalg.norm(b_cent - b_ax_comp, axis=1)

        crown_zone_threshold = root_proj_val + 0.65 * tooth_length_mm
        bone_filter = (b_perp_dist < 14.0) & (b_proj < crown_zone_threshold)
        filtered_bone = bone_world_all[bone_filter]

        if len(filtered_bone) < 20:
            # Fallback: relax perp constraint but keep projection filter
            bone_filter2  = (b_perp_dist < 22.0) & (b_proj < crown_zone_threshold)
            filtered_bone = bone_world_all[bone_filter2]
        if len(filtered_bone) == 0:
            continue

        # ── Find tooth voxels near the filtered alveolar bone ─────────────────
        local_tree  = cKDTree(filtered_bone)
        dists, _    = local_tree.query(tooth_world, k=1, workers=-1)
        iface_mask  = dists < 4.0
        if iface_mask.sum() < 5:
            continue
        iface_world = tooth_world[iface_mask]

        # ── Refine bone-crest position using a thin ±2 mm slice ──────────────
        # The mean of all interface voxels smears over the entire root face.
        # A narrow slice around the shallowest bone level gives the crest.
        rough_bone_centroid = iface_world.mean(axis=0)
        bone_level_proj     = float((rough_bone_centroid - centroid) @ long_axis)

        slice_mask = np.abs(proj - bone_level_proj) < 2.0
        if slice_mask.sum() < 8:
            slice_mask = np.abs(proj - bone_level_proj) < 4.0
        slice_world   = tooth_world[slice_mask] if slice_mask.sum() >= 5 else iface_world
        bone_centroid = slice_world.mean(axis=0)
        bone_proj     = float((bone_centroid - centroid) @ long_axis)

        # ── Crown / root tip, measurements ───────────────────────────────────
        crown_world      = tooth_world[np.argmax(proj)]
        root_world       = tooth_world[np.argmin(proj)]
        crown_proj       = float(proj.max())
        bone_to_crown_mm = float(crown_proj - bone_proj)
        crown_length_est = 0.40 * tooth_length_mm
        root_exposed_mm  = max(0.0, bone_to_crown_mm - crown_length_est)

        # ── Ring radius: perpendicular half-width of tooth at crest level ─────
        ref = np.array([0., 0., 1.])
        if abs(long_axis @ ref) > 0.9:
            ref = np.array([0., 1., 0.])
        u  = np.cross(long_axis, ref); u /= np.linalg.norm(u)
        v  = np.cross(long_axis, u)
        sc = slice_world - bone_centroid
        radii_slice = np.sqrt((sc @ u) ** 2 + (sc @ v) ** 2)
        ring_radius = float(np.percentile(radii_slice, 65))   # no inflation offset

        # ── Severity ─────────────────────────────────────────────────────────
        if   root_exposed_mm < 2.0: severity = 'normal'
        elif root_exposed_mm < 4.0: severity = 'mild'
        elif root_exposed_mm < 6.0: severity = 'moderate'
        else:                       severity = 'severe'

        results.append({
            'fdi':               int(lbl),
            'jaw':               FDI_TO_JAW.get(int(lbl), 'unknown'),
            'bone_point':        bone_centroid.tolist(),
            'crown_tip':         crown_world.tolist(),
            'root_tip':          root_world.tolist(),
            'long_axis':         long_axis.tolist(),
            'ring_radius_mm':    round(ring_radius, 1),
            'tooth_length_mm':   round(tooth_length_mm, 1),
            'crown_length_mm':   round(crown_length_est, 1),
            'bone_to_crown_mm':  round(bone_to_crown_mm, 1),
            'root_exposed_mm':   round(root_exposed_mm, 1),
            'severity':          severity,
        })

    counts = {s: sum(1 for r in results if r['severity'] == s)
              for s in ('normal', 'mild', 'moderate', 'severe')}
    return {'base': base, 'teeth': results, 'summary': counts}


# ── CBCT 2-D slice viewer ─────────────────────────────────────────────────────

_cbct_cache: dict = {}      # base → (data_f32, affine, vmin, vmax, axis_order)
_CBCT_CACHE_MAX = 2


def _encode_png_gray(arr_u8: np.ndarray) -> bytes:
    """Encode a 2-D uint8 HxW array as a lossless grayscale PNG (stdlib only)."""
    h, w = arr_u8.shape
    raw = b"".join(b"\x00" + arr_u8[y].tobytes() for y in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )


def _load_cbct(base: str):
    """Return (data, affine, vmin, vmax, axis_order) for the CBCT of *base*, or None."""
    if base in _cbct_cache:
        return _cbct_cache[base]

    cbct_path = ROOT / "output" / base / "cbct.nii.gz"
    if not cbct_path.exists():
        return None

    img  = nib.load(str(cbct_path))
    data = np.asarray(img.dataobj).astype(np.float32)

    # Auto-window from non-air voxels (HU > -500)
    fg = data[data > -500]
    vmin = float(np.percentile(fg, 1))  if len(fg) > 100 else float(data.min())
    vmax = float(np.percentile(fg, 99)) if len(fg) > 100 else float(data.max())

    # Map each anatomical plane to the most-aligned voxel axis
    aff3 = np.abs(img.affine[:3, :3])
    axis_order = {
        "sagittal": int(np.argmax(aff3[0])),  # world-X (L-R)
        "coronal":  int(np.argmax(aff3[1])),  # world-Y (A-P)
        "axial":    int(np.argmax(aff3[2])),  # world-Z (S-I)
    }

    entry = (data, img.affine, vmin, vmax, axis_order)
    if len(_cbct_cache) >= _CBCT_CACHE_MAX:
        del _cbct_cache[next(iter(_cbct_cache))]
    _cbct_cache[base] = entry
    return entry


def _slice_display_size(data_shape, axis_order, plane):
    """Return (w, h) of the PNG for *plane* after transpose."""
    ax  = axis_order[plane]
    rem = [data_shape[i] for i in range(3) if i != ax]   # [dim_a, dim_b]
    # After transpose: rows = dim_b, cols = dim_a  (we always put the 2nd dim as rows)
    # Exception: axial — rows = dim_a (A-P), cols = dim_b (L-R)
    if plane == "axial":
        return rem[1], rem[0]   # w=dim_b, h=dim_a
    else:
        return rem[0], rem[1]   # w=dim_a, h=dim_b (S-I becomes rows)


@app.get("/api/sliceinfo/{base}")
def api_sliceinfo(base: str):
    entry = _load_cbct(base)
    if entry is None:
        raise HTTPException(404, "cbct.nii.gz not found — re-run pipeline to generate it")
    data, _, vmin, vmax, axis_order = entry
    # Post-transpose pixel sizes for each plane so the frontend can set canvas dimensions
    sizes = {p: {"w": _slice_display_size(data.shape, axis_order, p)[0],
                 "h": _slice_display_size(data.shape, axis_order, p)[1]}
             for p in ("axial", "coronal", "sagittal")}
    return {"shape": list(data.shape), "axis_order": axis_order,
            "slice_sizes": sizes, "vmin": vmin, "vmax": vmax}


@app.get("/api/cbct_volume/{base}")
def api_cbct_volume(base: str):
    """Return the full uint8-windowed CBCT volume as raw bytes for client-side slicing.
    Shape and axis mapping are returned in response headers."""
    import json as _json
    entry = _load_cbct(base)
    if entry is None:
        raise HTTPException(404, "cbct.nii.gz not found — re-run pipeline to generate it")
    data, affine, vmin, vmax, axis_order = entry
    u8 = np.clip((data - vmin) / max(vmax - vmin, 1.0), 0.0, 1.0)
    u8 = (u8 * 255).astype(np.uint8)
    headers = {
        "X-Shape":      _json.dumps(list(data.shape)),
        "X-AxisOrder":  _json.dumps(axis_order),
        "X-ZSign":      str(float(affine[2, axis_order["axial"]])),
        "X-YSign":      str(float(affine[1, axis_order["coronal"]])),
        "Access-Control-Expose-Headers": "X-Shape, X-AxisOrder, X-ZSign, X-YSign",
    }
    return Response(content=np.ascontiguousarray(u8).tobytes(),
                    media_type="application/octet-stream", headers=headers)


@app.get("/api/slice/{base}/{axis}/{index}")
def api_slice(base: str, axis: str, index: int):
    entry = _load_cbct(base)
    if entry is None:
        raise HTTPException(404, "cbct.nii.gz not found")
    data, affine, vmin, vmax, axis_order = entry

    ax = axis_order.get(axis)
    if ax is None:
        raise HTTPException(422, f"axis must be axial/coronal/sagittal, got {axis!r}")

    index = max(0, min(data.shape[ax] - 1, index))
    sl = np.take(data, index, axis=ax).astype(np.float32)   # shape: two remaining dims

    # Transpose so the display axes are correct:
    #   coronal / sagittal → rows = S-I (K), cols = L-R or A-P
    #   axial              → rows = A-P (J), cols = L-R (I)
    sl = sl.T

    # Superior-up: for coronal/sagittal the row axis is K.
    # If affine maps K→+Z (positive sign), K=0 is inferior → flipud to put superior at top.
    ax_axial = axis_order["axial"]
    z_sign   = float(affine[2, ax_axial])
    if axis != "axial" and z_sign > 0:
        sl = np.flipud(sl)
    # For axial: rows = J (A-P). If affine maps J→+Y (positive), J=0 is posterior → flipud.
    if axis == "axial":
        ax_coronal = axis_order["coronal"]
        y_sign = float(affine[1, ax_coronal])
        if y_sign > 0:
            sl = np.flipud(sl)

    # Window → uint8
    sl_u8 = np.clip((sl - vmin) / max(vmax - vmin, 1.0), 0.0, 1.0)
    sl_u8 = (sl_u8 * 255).astype(np.uint8)

    return Response(content=_encode_png_gray(sl_u8), media_type="image/png")


# Static files & SPA root
app.mount("/static", StaticFiles(directory=str(UI_STATIC)), name="static")

@app.get("/")
def index():
    return FileResponse(str(UI_STATIC / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860)
