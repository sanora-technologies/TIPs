# ToothFairy3 Segmentation Pipeline

## Task

Automatically segment **78 anatomical structures** (77 foreground + background) from dental CBCT scans.
Dataset: 532 cases, classes include jawbones, all teeth, pulp chambers, IAC canals, maxillary sinuses, pharynx, and dental restorations.

---

## Architecture

**Framework:** nnUNet v2.7 — self-configuring deep learning segmentation framework  
**Model:** ResidualEncoderUNet-M (`nnUNetResEncUNetMPlans`)  
- Encoder: residual blocks with increasing channel depth  
- Decoder: skip connections from encoder to decoder  
- Deep supervision: loss computed at multiple resolution scales  
- Output: softmax over 78 classes per voxel  

**Training losses:** compound Dice + Cross-Entropy  
**Optimizer:** SGD with Nesterov momentum, polynomial LR decay over `num_epochs`  
**Augmentation:** rotation, scaling, elastic deformation, Gaussian noise/blur, brightness, contrast, mirroring

---

## Stage 1 — 3d_lowres (Coarse Segmentation)

**Why:** A single full-resolution pass over a large CBCT volume is too expensive and too hard to learn from scratch. The lowres model first establishes *where* each structure is.

**How:**
- Input: raw CT scan resampled to **0.454 mm** isotropic spacing
- Patch size: `[64, 160, 160]` voxels
- Input channels: **1** (CT intensity only)
- Output: segmentation map with 78 classes

**Training:** 1073 epochs, batch size 2, `oversample_foreground=0.85`  
**Result:** EMA pseudo-Dice **0.714** — solid coarse predictions across all 78 classes

**Limitation:** Lower resolution means blurry boundaries, missed fine structures (thin pulp canals, incisive canals). Good enough to know *where*, not precise enough for clinical use alone.

---

## Stage 1.5 — Pseudo-Label Generation

**Why:** Stage 2 needs to see Stage 1's predictions *on the training data* at training time — not just the ground-truth labels. This teaches Stage 2 to correct realistic Stage 1 errors.

**How:**
1. Run Stage 1 inference on all 532 training cases
2. Save predictions as `.b2nd` compressed arrays (`predicted_next_stage/`)
3. These become Stage 2's second input channel group during training

**Result:** 532 pseudo-label files generated. Stage 2 sees both the ground-truth segmentation (for loss computation) and Stage 1's imperfect prediction (as input context).

---

## Stage 2 — 3d_cascade_fullres (Full-Resolution Refinement)

**Why:** Stage 2 knows where structures are (from Stage 1) and can focus on getting boundaries, fine structures, and rare classes right at full resolution.

**How:**
- Input: CT scan at **0.300 mm** isotropic spacing + Stage 1 prediction expanded to **77-channel one-hot map**
- Total input channels: **78** (1 CT + 77 binary masks, one per foreground class)
- Patch size: `[64, 160, 160]` voxels (same shape, finer spacing = more detail)
- The one-hot expansion is done **on GPU** at training time (not in the dataloader), keeping RAM usage low

**Cascade augmentations** (applied to Stage 1 prediction before feeding to Stage 2):
- **Morphological** (dilate/erode): teaches Stage 2 to handle Stage 1 boundaries that are slightly too large or too small
- **Connected-component removal**: teaches Stage 2 to handle cases where Stage 1 missed entire tooth instances

These augmentations simulate realistic Stage 1 failure modes. Without them, Stage 2 would overfit to perfect Stage 1 predictions and fail on real errors at inference time.

**Training:** 2000 epochs target, batch size 1 (VRAM limit at 78 channels), `oversample_foreground=0.85`  
**Best checkpoint:** epoch ~235, EMA pseudo-Dice **0.699** (exceeded Stage 1's epoch-240 score of 0.580)

---

## Inference Pipeline

```
Raw CBCT scan
      │
      ▼
Stage 1 (3d_lowres)
  → coarse 78-class segmentation at 0.454 mm
      │
      ▼
Stage 2 (3d_cascade_fullres)
  → CT (0.300 mm) + Stage 1 prediction (77-channel one-hot)
  → refined 78-class segmentation at full resolution
      │
      ▼
Final segmentation
```

**Checkpoint used:** `checkpoint_best.pth` in each stage's `fold_0/` folder (highest EMA Dice seen during training)  
**Test-time augmentation:** full mirroring TTA (8 passes) for maximum accuracy  
**Approximate inference time:** 10–25 min per scan on RTX 5060 Ti

---

## Why Cascade Beats Single-Stage

| | Single-stage fullres | Cascade (Stage 1 + 2) |
|---|---|---|
| Learns structure location | Hard — large volume, many classes | Easy — Stage 1 handles this |
| Fine boundary detail | Limited by patch coverage | Stage 2 focuses here |
| Rare/small classes | Often missed | Stage 1 hint guides Stage 2 |
| Robust to Stage 1 errors | N/A | Trained on augmented (imperfect) Stage 1 predictions |

Stage 2 at epoch 240 already matched Stage 1's *final* performance — the cascade context (77 extra input channels) accelerates learning because the model doesn't need to rediscover structure locations from scratch.

---

## Class Performance Summary (Stage 2, epoch ~353)

| Group | Classes | Avg Dice |
|---|---|---|
| Jawbones, sinuses | 1, 2, 5, 6 | 0.75–0.97 |
| Teeth (11–42) | 32 teeth | ~0.75 |
| IAC canals | 3, 4, 43, 44 | 0.37–0.65 |
| Pulp chambers | 46–77 | 0.47–0.75 |

IAC canals and pulp chambers are inherently hard: 1–3 voxels wide at 0.3 mm spacing. These scores represent state-of-the-art performance for this resolution and class count.
