"""
Analysis overlay drawing: reference planes, angle arcs, linear arrows.
All drawing is on BGR numpy arrays (cv2 convention); convert before/after.
"""
import cv2
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────────
_SN    = (0, 220, 220)    # yellow-cyan : SN plane
_FH    = (0, 200, 255)    # yellow      : Frankfort horizontal
_MP    = (255, 80, 200)   # magenta     : mandibular plane
_NA    = (0, 140, 255)    # orange      : NA line
_NB    = (255, 140, 0)    # blue        : NB line
_U1    = (80, 230, 80)    # green       : upper incisor axis
_L1    = (80, 80, 230)    # red         : lower incisor axis
_MEAS  = (0, 255, 220)    # cyan        : linear arrows
_NPERP = (160, 160, 160)  # grey        : N-perpendicular


# ── geometry helpers ─────────────────────────────────────────────────────────

def _pt(p):
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def _clip_line_to_canvas(p1, p2, H, W):
    """Cohen-Sutherland clip; returns None if line fully outside."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    def code(x, y):
        c = 0
        if x < 0:     c |= 1
        elif x > W-1: c |= 2
        if y < 0:     c |= 4
        elif y > H-1: c |= 8
        return c

    c1, c2 = code(x1, y1), code(x2, y2)
    for _ in range(10):
        if not (c1 | c2):
            return (int(x1), int(y1)), (int(x2), int(y2))
        if c1 & c2:
            return None
        c = c1 if c1 else c2
        if c & 8:
            x = x1 + (x2-x1) * (H-1-y1) / (y2-y1+1e-9)
            y = H-1
        elif c & 4:
            x = x1 + (x2-x1) * (0-y1) / (y2-y1+1e-9)
            y = 0
        elif c & 2:
            y = y1 + (y2-y1) * (W-1-x1) / (x2-x1+1e-9)
            x = W-1
        else:
            y = y1 + (y2-y1) * (0-x1) / (x2-x1+1e-9)
            x = 0
        if c == c1:
            x1, y1, c1 = x, y, code(x, y)
        else:
            x2, y2, c2 = x, y, code(x, y)
    return None


def _extend(p1, p2, H, W, factor=0.6):
    """Extend line beyond both endpoints then clip to canvas."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    ep1 = p1 - d * factor
    ep2 = p2 + d * factor
    result = _clip_line_to_canvas(ep1, ep2, H, W)
    if result is None:
        return _pt(p1), _pt(p2)
    return result


def _line_intersection(p1, p2, p3, p4):
    """Intersection of infinite lines p1-p2 and p3-p4. Returns None if parallel."""
    d1 = np.asarray(p2, float) - np.asarray(p1, float)
    d2 = np.asarray(p4, float) - np.asarray(p3, float)
    cross = d1[0]*d2[1] - d1[1]*d2[0]
    if abs(cross) < 1e-6:
        return None
    t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cross
    return np.asarray(p1, float) + t * d1


def _anterior_unit(lm):
    if "S" in lm and "N" in lm:
        v = np.asarray(lm["N"], float) - np.asarray(lm["S"], float)
        n = np.linalg.norm(v)
        if n > 1:
            return v / n
    return np.array([-1.0, 0.0])


def _fh_perp_direction(lm, H, W):
    """Unit vector perpendicular to FH pointing inferiorly (y-down)."""
    if "Or" in lm and "Po" in lm:
        fh = np.asarray(lm["Po"], float) - np.asarray(lm["Or"], float)
        n = np.linalg.norm(fh)
        if n > 1:
            fh /= n
            perp = np.array([-fh[1], fh[0]])   # rotate 90 deg
            if perp[1] < 0:
                perp = -perp               # ensure it points down (inferior)
            return perp
    return np.array([0.0, 1.0])


# ── drawing primitives ────────────────────────────────────────────────────────

def _draw_line(img, p1, p2, color, thick, extend=False):
    H, W = img.shape[:2]
    if extend:
        result = _extend(p1, p2, H, W)
    else:
        result = _pt(p1), _pt(p2)
    cv2.line(img, result[0], result[1], color, thick, cv2.LINE_AA)


def _label(img, pt, text, color, fs=None, offset=(6, -4)):
    H, W = img.shape[:2]
    fs = fs or max(0.35, min(H, W) / 2200)
    thick = max(1, int(fs * 2))
    x = int(pt[0]) + offset[0]
    y = int(pt[1]) + offset[1]
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (0, 0, 0), thick+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                fs, color, thick, cv2.LINE_AA)


def _arrow(img, p1, p2, color, thick):
    cv2.arrowedLine(img, _pt(p1), _pt(p2), color, thick,
                    cv2.LINE_AA, tipLength=0.2)
    cv2.arrowedLine(img, _pt(p2), _pt(p1), color, thick,
                    cv2.LINE_AA, tipLength=0.2)


def _arc(img, vertex, p1, p2, radius, color, thick):
    """Small arc at vertex between rays toward p1 and p2."""
    v = np.asarray(vertex, float)
    a1 = float(np.degrees(np.arctan2(p1[1]-v[1], p1[0]-v[0])))
    a2 = float(np.degrees(np.arctan2(p2[1]-v[1], p2[0]-v[0])))
    lo, hi = sorted([a1, a2])
    if hi - lo > 180:               # take the smaller arc
        lo, hi = hi, lo + 360
    cv2.ellipse(img, _pt(v), (int(radius), int(radius)),
                0, lo, hi, color, thick, cv2.LINE_AA)


def _dot(img, pt, color, r):
    cv2.circle(img, _pt(pt), r+1, (0, 0, 0), -1)
    cv2.circle(img, _pt(pt), r, color, -1)


# ── public entry point ────────────────────────────────────────────────────────

def draw_analysis(img_rgb: np.ndarray, lm: dict, tier: int) -> np.ndarray:
    """Draw analysis overlay on a copy of img_rgb. Returns annotated RGB image."""
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    thick = max(1, min(H, W) // 600)
    arc_r = max(18, min(H, W) // 35)
    has = lambda *k: all(k_ in lm for k_ in k)
    ant = _anterior_unit(lm)

    # ── Tier 1 ───────────────────────────────────────────────────────────────

    # SN plane
    if has("S", "N"):
        _draw_line(img, lm["S"], lm["N"], _SN, thick, extend=True)
        _label(img, lm["N"], "SN", _SN, offset=(6, -6))

    # Frankfort Horizontal
    if has("Or", "Po"):
        _draw_line(img, lm["Or"], lm["Po"], _FH, thick, extend=True)
        _label(img, lm["Po"], "FH", _FH, offset=(-24, -6))

    # Mandibular plane
    if has("Go", "Me"):
        _draw_line(img, lm["Go"], lm["Me"], _MP, thick, extend=True)
        _label(img, lm["Me"], "MP", _MP, offset=(6, 14))

    # NA line
    if has("N", "A"):
        _draw_line(img, lm["N"], lm["A"], _NA, thick, extend=True)

    # NB line
    if has("N", "B"):
        _draw_line(img, lm["N"], lm["B"], _NB, thick, extend=True)

    # SNA arc at N (between rays N→S and N→A)
    if has("S", "N", "A"):
        _arc(img, lm["N"], lm["S"], lm["A"], arc_r, _SN, thick)

    # SNB arc at N — slightly larger so the two arcs don't overlap
    if has("S", "N", "B"):
        _arc(img, lm["N"], lm["S"], lm["B"], int(arc_r * 1.45), _NB, thick)

    # FMA arc — draw at the intersection of FH and MP lines (correct location)
    if has("Or", "Po", "Go", "Me"):
        inter = _line_intersection(lm["Or"], lm["Po"], lm["Go"], lm["Me"])
        if inter is not None:
            # rays from intersection toward Or (FH) and toward Me (MP)
            _arc(img, inter, lm["Or"], lm["Me"], arc_r, _FH, thick)
            _label(img, inter, "FMA", _FH, offset=(arc_r + 2, 4))

    # Upper incisor axis
    if has("UIA", "UIT"):
        _draw_line(img, lm["UIA"], lm["UIT"], _U1, thick+1)
        _dot(img, lm["UIT"], _U1, max(3, thick+1))

    # Lower incisor axis
    if has("LIA", "LIT"):
        _draw_line(img, lm["LIA"], lm["LIT"], _L1, thick+1)
        _dot(img, lm["LIT"], _L1, max(3, thick+1))

    # Overjet arrow (along anterior direction at incisor height)
    if has("UIT", "LIT"):
        mid_y = (lm["UIT"][1] + lm["LIT"][1]) / 2
        # project UIT and LIT onto the perpendicular-to-ant axis at mid_y
        perp = np.array([-ant[1], ant[0]])  # perpendicular to ant
        oj_base = mid_y                      # y level for the arrow
        # draw arrow in the ant direction between the two points
        oj_p1 = np.array([lm["UIT"][0], oj_base])
        oj_p2 = np.array([lm["LIT"][0], oj_base])
        _arrow(img, oj_p1, oj_p2, _MEAS, thick)
        mid_pt = (oj_p1 + oj_p2) / 2
        _label(img, mid_pt, "OJ", _MEAS, offset=(-8, -6))

    # Overbite arrow (vertical between UIT and LIT)
    if has("UIT", "LIT"):
        ob_x = min(lm["UIT"][0], lm["LIT"][0]) - 14
        _arrow(img, (ob_x, lm["UIT"][1]), (ob_x, lm["LIT"][1]), _MEAS, thick)
        mid_y = (lm["UIT"][1] + lm["LIT"][1]) / 2
        _label(img, (ob_x, mid_y), "OB", _MEAS, offset=(-22, 4))

    # Key skeletal dots
    for sym, col in [("S", _SN), ("N", _SN), ("A", _NA), ("B", _NB),
                     ("Go", _MP), ("Me", _MP), ("Or", _FH), ("Po", _FH)]:
        if sym in lm:
            _dot(img, lm[sym], col, max(3, thick+1))

    # ── Tier 2 additional overlays ────────────────────────────────────────────

    if tier >= 2:
        # N-perpendicular: line through N perpendicular to FH
        if has("N"):
            perp_dir = _fh_perp_direction(lm, H, W)
            scale = max(H, W)
            np1 = np.asarray(lm["N"], float) - perp_dir * scale
            np2 = np.asarray(lm["N"], float) + perp_dir * scale
            result = _clip_line_to_canvas(np1, np2, H, W)
            if result:
                cv2.line(img, result[0], result[1], _NPERP, thick, cv2.LINE_AA)
            _label(img, lm["N"], "N-perp", _NPERP, offset=(6, 16))

        # Anterior facial height: N to Me
        if has("N", "Me"):
            x_afh = int(min(lm["N"][0], lm["Me"][0])) - 22
            _arrow(img, (x_afh, lm["N"][1]), (x_afh, lm["Me"][1]), _MEAS, thick)
            mid_y = (lm["N"][1] + lm["Me"][1]) / 2
            _label(img, (x_afh, mid_y), "AFH", _MEAS, offset=(-34, 4))

        # Lower anterior facial height: ANS to Me
        if has("ANS", "Me"):
            x_lafh = int(lm["ANS"][0]) - 8
            _arrow(img, (x_lafh, lm["ANS"][1]), (x_lafh, lm["Me"][1]), _MEAS, thick)
            mid_y = (lm["ANS"][1] + lm["Me"][1]) / 2
            _label(img, (x_lafh, mid_y), "LAFH", _MEAS, offset=(-40, 4))

        # E-line: Pn to Pog`
        if has("Pn", "Pog`"):
            _draw_line(img, lm["Pn"], lm["Pog`"], (120, 230, 120), thick, extend=True)
            _label(img, lm["Pn"], "E-line", (120, 230, 120), offset=(6, -4))

        # Interincisal arc at UIT
        if has("UIA", "UIT", "LIA", "LIT"):
            _arc(img, lm["UIT"], lm["UIA"], lm["LIA"], arc_r, _U1, thick)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
