"""
Cephalometric measurements from 29 landmarks.
All coordinates are (x, y) pixel arrays in original image space.
px_mm: mm per pixel for physical unit conversion.
"""
import numpy as np


# ── geometry helpers ─────────────────────────────────────────────────────────

def _v(a, b):
    return np.asarray(b, float) - np.asarray(a, float)


def _angle_at(vertex, p1, p2) -> float:
    v1, v2 = _v(vertex, p1), _v(vertex, p2)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _angle_lines(a, b, c, d) -> float:
    """Acute angle between line a-b and line c-d."""
    v1, v2 = _v(a, b), _v(c, d)
    cos_a = abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, 0.0, 1.0))))


def _anterior_unit(lm: dict):
    """
    Unit vector pointing in the anatomically anterior direction.
    Derived from S→N (sella to nasion always points anterior-superior).
    Falls back to (-1, 0) if landmarks missing.
    """
    if "S" in lm and "N" in lm:
        v = _v(lm["S"], lm["N"])
        n = np.linalg.norm(v)
        if n > 1:
            return v / n
    return np.array([-1.0, 0.0])


def _fh_unit(lm: dict):
    """
    Unit vector along Frankfort Horizontal (Or → Po direction,
    re-signed so it points posteriorly — we want anterior = +proj).
    Returns (anterior_unit, inferior_unit) both perpendicular to each other.
    """
    if "Or" in lm and "Po" in lm:
        # Or is anterior, Po is posterior → Or→Po is the posterior direction
        # We want the anterior direction = Po → Or
        v = _v(lm["Po"], lm["Or"])   # points anterior
        n = np.linalg.norm(v)
        if n > 1:
            fh_ant = v / n
            # inferior perpendicular (y increases downward in image coords)
            fh_inf = np.array([-fh_ant[1], fh_ant[0]])
            if fh_inf[1] < 0:          # flip so inferior points down
                fh_inf = -fh_inf
            return fh_ant, fh_inf
    ant = _anterior_unit(lm)
    inf = np.array([-ant[1], ant[0]])
    if inf[1] < 0:
        inf = -inf
    return ant, inf


def _status(val, lo, hi):
    """Return (status_str, direction_str)."""
    if lo <= val <= hi:
        return "Normal", ""
    half = (hi - lo) / 2
    if val > hi:
        return ("Mild" if val - hi <= half else "Severe"), "High"
    return ("Mild" if lo - val <= half else "Severe"), "Low"


# ── tier 1 ───────────────────────────────────────────────────────────────────

def compute_tier1(lm: dict, px_mm: float) -> list[dict]:
    rows = []
    has = lambda *k: all(k_ in lm for k_ in k)
    ant, inf = _fh_unit(lm)

    sna = snb = None

    if has("S", "N", "A"):
        sna = _angle_at(lm["N"], lm["S"], lm["A"])
        st, d = _status(sna, 80, 84)
        rows.append({"name": "SNA", "value": sna, "unit": "°",
                     "normal": "80-84", "status": st, "note": d})

    if has("S", "N", "B"):
        snb = _angle_at(lm["N"], lm["S"], lm["B"])
        st, d = _status(snb, 78, 82)
        rows.append({"name": "SNB", "value": snb, "unit": "°",
                     "normal": "78-82", "status": st, "note": d})

    if sna is not None and snb is not None:
        anb = sna - snb
        st, d = _status(anb, 0, 4)
        rows.append({"name": "ANB", "value": anb, "unit": "°",
                     "normal": "0-4", "status": st, "note": d})
        cls = "Class III" if anb < 0 else ("Class I" if anb <= 4 else "Class II")
        rows.append({"name": "Skeletal Class", "value": anb, "unit": "°",
                     "normal": "0-4", "status": cls, "note": ""})

    if has("Or", "Po", "Go", "Me"):
        fma = _angle_lines(lm["Or"], lm["Po"], lm["Go"], lm["Me"])
        st, d = _status(fma, 22, 28)
        rows.append({"name": "FMA", "value": fma, "unit": "°",
                     "normal": "22-28", "status": st, "note": d})

    if has("S", "N", "Go", "Me"):
        snmp = _angle_lines(lm["S"], lm["N"], lm["Go"], lm["Me"])
        st, d = _status(snmp, 28, 36)
        rows.append({"name": "SN-MP", "value": snmp, "unit": "°",
                     "normal": "28-36", "status": st, "note": d})

    if has("UIA", "UIT", "S", "N"):
        u1sn = _angle_lines(lm["UIA"], lm["UIT"], lm["S"], lm["N"])
        st, d = _status(u1sn, 98, 110)
        rows.append({"name": "U1-SN", "value": u1sn, "unit": "°",
                     "normal": "98-110", "status": st, "note": d})

    if has("LIA", "LIT", "Go", "Me"):
        impa = _angle_lines(lm["LIA"], lm["LIT"], lm["Go"], lm["Me"])
        st, d = _status(impa, 85, 95)
        rows.append({"name": "IMPA (L1-MP)", "value": impa, "unit": "°",
                     "normal": "85-95", "status": st, "note": d})

    if has("UIT", "LIT"):
        # Project UIT-LIT onto the anterior direction.
        # Positive = upper incisor is in front of lower = normal overjet.
        oj = float(np.dot(_v(lm["LIT"], lm["UIT"]), ant)) * px_mm
        st, d = _status(oj, 1, 4)
        rows.append({"name": "Overjet", "value": oj, "unit": "mm",
                     "normal": "1-4", "status": st, "note": d})

        # Overbite: UIT above LIT → positive (y increases downward in image).
        ob = float(np.dot(_v(lm["UIT"], lm["LIT"]), inf)) * px_mm
        st, d = _status(ob, 1, 4)
        rows.append({"name": "Overbite", "value": ob, "unit": "mm",
                     "normal": "1-4", "status": st, "note": d})

    return rows


# ── tier 2 ───────────────────────────────────────────────────────────────────

def compute_tier2(lm: dict, px_mm: float) -> list[dict]:
    rows = []
    has = lambda *k: all(k_ in lm for k_ in k)
    ant, _ = _fh_unit(lm)

    if has("UIA", "UIT", "LIA", "LIT"):
        # Angle between the two incisor long axes (tip-to-apex vectors pointing away from each other)
        v_up = _v(lm["UIT"], lm["UIA"])   # upper: tip → apex
        v_lo = _v(lm["LIT"], lm["LIA"])   # lower: tip → apex
        cos_a = np.dot(v_up, v_lo) / (np.linalg.norm(v_up) * np.linalg.norm(v_lo) + 1e-9)
        inter = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        st, d = _status(inter, 125, 135)
        rows.append({"name": "Interincisal Angle", "value": inter, "unit": "°",
                     "normal": "125-135", "status": st, "note": d})

    if has("N", "A", "Pog"):
        convex = _angle_at(lm["A"], lm["N"], lm["Pog"])
        st, d = _status(convex, 165, 175)
        rows.append({"name": "Facial Convexity (N-A-Pog)", "value": convex, "unit": "°",
                     "normal": "165-175", "status": st, "note": d})

    # N-perp distances: project onto the anterior (FH) direction.
    # Positive = point is anterior to N.
    if has("N", "A"):
        a_nperp = float(np.dot(_v(lm["N"], lm["A"]), ant)) * px_mm
        st, d = _status(a_nperp, -1, 1)
        rows.append({"name": "A to N-Perp", "value": a_nperp, "unit": "mm",
                     "normal": "-1 to 1", "status": st, "note": d})

    if has("N", "Pog"):
        pog_nperp = float(np.dot(_v(lm["N"], lm["Pog"]), ant)) * px_mm
        st, d = _status(pog_nperp, -4, 0)
        rows.append({"name": "Pog to N-Perp", "value": pog_nperp, "unit": "mm",
                     "normal": "-4 to 0", "status": st, "note": d})

    if has("N", "Me"):
        afh = abs(lm["Me"][1] - lm["N"][1]) * px_mm
        st, d = _status(afh, 110, 130)
        rows.append({"name": "Ant. Facial Height (N-Me)", "value": afh, "unit": "mm",
                     "normal": "110-130", "status": st, "note": d})

    if has("ANS", "Me"):
        lafh = abs(lm["Me"][1] - lm["ANS"][1]) * px_mm
        st, d = _status(lafh, 60, 72)
        rows.append({"name": "Lower Facial Height (ANS-Me)", "value": lafh, "unit": "mm",
                     "normal": "60-72", "status": st, "note": d})

    if has("Ls", "Li", "Pn", "Pog`"):
        e_line = _v(lm["Pn"], lm["Pog`"])
        e_norm = np.linalg.norm(e_line)
        if e_norm > 1:
            e_hat = e_line / e_norm
            # normal pointing anteriorly (into the face), sign by ant direction
            n_hat = np.array([-e_hat[1], e_hat[0]])
            if np.dot(n_hat, ant) < 0:
                n_hat = -n_hat
            ls_dist = float(np.dot(_v(lm["Pn"], lm["Ls"]), n_hat)) * px_mm
            li_dist = float(np.dot(_v(lm["Pn"], lm["Li"]), n_hat)) * px_mm
            st, d = _status(ls_dist, -4, -2)
            rows.append({"name": "Ls to E-Line", "value": ls_dist, "unit": "mm",
                         "normal": "-4 to -2", "status": st, "note": d})
            st, d = _status(li_dist, -2, 0)
            rows.append({"name": "Li to E-Line", "value": li_dist, "unit": "mm",
                         "normal": "-2 to 0", "status": st, "note": d})

    return rows


# ── public API ────────────────────────────────────────────────────────────────

def get_measurements(lm: dict, px_mm: float, tier: int) -> list[dict]:
    t1 = compute_tier1(lm, px_mm)
    if tier == 1:
        return t1
    return t1 + compute_tier2(lm, px_mm)


def to_table_rows(measurements: list[dict]) -> list[list]:
    STATUS_ICON = {"Normal": "✓", "Mild": "!", "Severe": "X",
                   "Class I": "*", "Class II": "*", "Class III": "*"}
    rows = []
    for m in measurements:
        icon = STATUS_ICON.get(m["status"], "")
        val_str = f"{m['value']:.1f}{m['unit']}"
        note = m["note"] if m["note"] else "-"
        rows.append([m["name"], val_str, m["normal"],
                     f"{icon} {m['status']}", note])
    return rows
