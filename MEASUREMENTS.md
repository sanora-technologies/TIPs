# Cephalometric Measurements

Model: HRNet-W32, 29 landmarks, trained on Aariz/CEPHA29 dataset.  
Checkpoint: `checkpoints/stage2/best.pt`

---

## Coordinate conventions

- Image space: `(x, y)` pixels, origin top-left, y increases downward.
- **Anterior direction** is derived from `S → N` (Sella to Nasion), not assumed from image orientation. This makes all signed measurements work regardless of whether the face looks left or right.
- **FH direction** is derived from `Or → Po` (Orbitale to Porion). Used for N-perpendicular and projected linear distances.
- Physical distances: multiply pixel distance by `px_mm` (machine-specific, 0.089–0.144 mm/px).

---

## Landmark symbols used

| Symbol | Name |
|--------|------|
| S | Sella |
| N | Nasion |
| A | A-point (maxilla) |
| B | B-point (mandible) |
| Or | Orbitale |
| Po | Porion |
| Go | Gonion |
| Me | Menton |
| Gn | Gnathion |
| ANS | Anterior Nasal Spine |
| UIT / UIA | Upper Incisor Tip / Apex |
| LIT / LIA | Lower Incisor Tip / Apex |
| Pog | Pogonion |
| Pn | Pronasale |
| Pog` | Soft Tissue Pogonion |
| Ls / Li | Labrale Superius / Inferius |

---

## Tier 1 — Core measurements

All implemented in `src/measurements.py → compute_tier1()`.

### Angular

| Name | Landmarks | How computed | Normal |
|------|-----------|--------------|--------|
| SNA | S, N, A | `angle_at_vertex(N, S, A)` | 80–84° |
| SNB | S, N, B | `angle_at_vertex(N, S, B)` | 78–82° |
| ANB | — | `SNA − SNB` | 0–4° |
| FMA | Or, Po, Go, Me | Acute angle between lines Or-Po and Go-Me | 22–28° |
| SN-MP | S, N, Go, Me | Acute angle between lines S-N and Go-Me | 28–36° |
| U1-SN | UIA, UIT, S, N | Acute angle between incisor axis (UIA→UIT) and SN line | 98–110° |
| IMPA | LIA, LIT, Go, Me | Acute angle between lower incisor axis and mandibular plane | 85–95° |

`angle_at_vertex(V, P1, P2)` = angle in degrees at V using dot product of vectors V→P1 and V→P2.  
`acute_angle(line1, line2)` = always returns ≤ 90° using `abs(dot) / (|v1||v2|)`.

### Skeletal classification

Derived from ANB:
- ANB < 0° → Class III  
- 0–4° → Class I  
- > 4° → Class II

### Linear

| Name | How computed | Normal |
|------|--------------|--------|
| Overjet | `dot(UIT − LIT, anterior_unit) × px_mm` | 1–4 mm |
| Overbite | `dot(LIT − UIT, inferior_unit) × px_mm` | 1–4 mm |

`anterior_unit` = `normalize(N − S)`.  
`inferior_unit` = `anterior_unit` rotated 90° clockwise (points down in image).  
Positive overjet = upper incisor is in front of lower. Positive overbite = upper incisor is above lower.

---

## Tier 2 — Extended measurements

Implemented in `src/measurements.py → compute_tier2()`.

| Name | Landmarks | How computed | Normal |
|------|-----------|--------------|--------|
| Interincisal Angle | UIA, UIT, LIA, LIT | Angle between vectors (UIT→UIA) and (LIT→LIA) | 125–135° |
| Facial Convexity | N, A, Pog | `angle_at_vertex(A, N, Pog)` | 165–175° |
| A to N-Perp | N, A | `dot(A − N, fh_anterior_unit) × px_mm` | −1 to 1 mm |
| Pog to N-Perp | N, Pog | `dot(Pog − N, fh_anterior_unit) × px_mm` | −4 to 0 mm |
| Ant. Facial Height | N, Me | `|Me.y − N.y| × px_mm` | 110–130 mm |
| Lower Facial Height | ANS, Me | `|Me.y − ANS.y| × px_mm` | 60–72 mm |
| Ls to E-Line | Ls, Pn, Pog` | Signed distance from Ls to line Pn–Pog`, positive = anterior | −4 to −2 mm |
| Li to E-Line | Li, Pn, Pog` | Same for Li | −2 to 0 mm |

**N-Perp** = line through N perpendicular to FH (Or–Po). A/Pog distance is their projection onto the FH direction.

`fh_anterior_unit` = `normalize(Or − Po)` (points from Po toward Or = anteriorly).

---

## Status flags

Each measurement returns:

```python
{
    "name": str,
    "value": float,       # numeric result
    "unit": "°" or "mm",
    "normal": str,        # e.g. "80-84"
    "status": str,        # "Normal" | "Mild" | "Severe" | "Class I/II/III"
    "note": str           # "High" | "Low" | ""
}
```

Thresholds: deviation within half the normal range width = Mild; beyond = Severe.

---

## Image overlay (`src/draw_analysis.py`)

`draw_analysis(img_rgb, lm_dict, tier)` — takes RGB numpy array, returns RGB numpy array.

| Overlay element | Color | Tier |
|----------------|-------|------|
| SN plane | Cyan | 1+ |
| Frankfort Horizontal | Yellow | 1+ |
| Mandibular plane | Magenta | 1+ |
| NA / NB lines | Orange / Blue | 1+ |
| Upper / Lower incisor axis | Green / Red | 1+ |
| SNA, SNB arcs at N | Cyan / Blue | 1+ |
| FMA arc at FH–MP intersection | Yellow | 1+ |
| Overjet / Overbite arrows | Cyan | 1+ |
| N-perpendicular (perp to FH) | Grey | 2 |
| AFH / LAFH arrows | Cyan | 2 |
| E-line | Light green | 2 |
| Interincisal arc at UIT | Green | 2 |

---

## Minimal inference snippet

```python
import torch, numpy as np, cv2
from src.model import CephalometricModel, soft_argmax
from src.measurements import get_measurements, to_table_rows

LANDMARK_ORDER = [
    "A","ANS","B","Me","N","Or","Pog","PNS","Pn","R","S",
    "Ar","Co","Gn","Go","Po","LPM","LIT","LMT","UPM","UIA",
    "UIT","UMT","LIA","Li","Ls","N`","Pog`","Sn",
]

model = CephalometricModel(pretrained=False, use_cvm_head=False, heatmap_stride=2)
ckpt  = torch.load("checkpoints/stage2/best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
if ckpt.get("ema_shadow"):
    state = model.state_dict()
    for k, v in ckpt["ema_shadow"].items():
        if k in state: state[k] = v
    model.load_state_dict(state)
model.eval()

# --- preprocess your image to a (1,3,640,640) tensor (see app.py:_preprocess) ---
with torch.no_grad():
    hm = model(tensor)["heatmaps"]          # (1, 29, 320, 320)
coords = soft_argmax(hm)[0].numpy() * 2     # (29, 2) in 640x640 space
# --- map back to original pixels (see app.py:_to_original_space) ---

lm = {sym: coords[i] for i, sym in enumerate(LANDMARK_ORDER)}
px_mm = 0.100   # replace with actual machine value
rows = get_measurements(lm, px_mm, tier=1)   # or tier=2
print(to_table_rows(rows))
```
