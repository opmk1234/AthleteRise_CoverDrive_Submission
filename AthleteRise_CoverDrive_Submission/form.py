# form.py
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    PL = mp.solutions.pose.PoseLandmark
except Exception:
    MP_AVAILABLE = False
    PL = None  # We'll fall back to integer indices if names aren't available.


# ---- Landmark name -> index map (MediaPipe 33 keypoints) ----
# If mediapipe is available we’ll fetch indices from PoseLandmark; else we define a static map.
STATIC_IDX = {
    "NOSE": 0, "LEFT_EYE_INNER": 1, "LEFT_EYE": 2, "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "LEFT_MOUTH": 9, "RIGHT_MOUTH": 10,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17, "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19, "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21, "RIGHT_THUMB": 22,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}
def LMI(name: str) -> int:
    if MP_AVAILABLE and hasattr(PL, name):
        return getattr(PL, name).value
    return STATIC_IDX[name]


# ---- Helpers to read keypoints regardless of format ----
def _is_norm_coord(x: float, y: float) -> bool:
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

def _normalize_if_needed(pt: Tuple[float, float],
                         img_shape: Optional[Tuple[int, int, int]]) -> Tuple[float, float]:
    """If coords look normalized and we know image shape, scale to pixels; else return as-is."""
    if pt is None:
        return None
    x, y = pt
    if img_shape is not None and _is_norm_coord(x, y):
        h, w = img_shape[:2]
        return (x * w, y * h)
    return (x, y)

def _extract_point(keypoints: Any, idx_or_name: Any,
                   img_shape: Optional[Tuple[int, int, int]]) -> Optional[Tuple[float, float]]:
    """
    Supports:
      - dict{name: (x,y)} or dict[index]: (x,y)
      - list/tuple of length 33 with (x,y) or (x,y,conf) per index
      - list of items like (idx, x, y) or [idx, x, y]
    Returns pixel-space coords if possible; otherwise normalized/pixel as provided.
    """
    # Resolve index from name if needed
    idx = None
    if isinstance(idx_or_name, str):
        try:
            idx = LMI(idx_or_name)
        except Exception:
            return None
    else:
        idx = int(idx_or_name)

    # Dict by name or index
    if isinstance(keypoints, dict):
        if isinstance(idx_or_name, str) and idx_or_name in keypoints:
            pt = keypoints[idx_or_name]
            return _normalize_if_needed((pt[0], pt[1]), img_shape)
        if idx in keypoints:
            pt = keypoints[idx]
            return _normalize_if_needed((pt[0], pt[1]), img_shape)

    # Sequence by index
    if isinstance(keypoints, (list, tuple)):
        # If looks like flat list of 33 entries
        if len(keypoints) > idx:
            item = keypoints[idx]
            # item could be (x,y), (x,y,score), or (id,x,y)
            if isinstance(item, (list, tuple)):
                if len(item) >= 3 and isinstance(item[0], (int, float)) and isinstance(item[1], (int, float)) and isinstance(item[2], (int, float)):
                    # Ambiguous: could be (x,y,conf) OR (id,x,y). Try to detect id==idx
                    if len(item) == 3 and int(item[0]) == idx:
                        pt = (float(item[1]), float(item[2]))
                    else:
                        pt = (float(item[0]), float(item[1]))
                elif len(item) >= 2:
                    pt = (float(item[0]), float(item[1]))
                else:
                    pt = None
                return _normalize_if_needed(pt, img_shape)
        # Or list of (id,x,y) triplets (unordered)
        try:
            for it in keypoints:
                if isinstance(it, (list, tuple)) and len(it) >= 3:
                    i = int(it[0])
                    if i == idx:
                        pt = (float(it[1]), float(it[2]))
                        return _normalize_if_needed(pt, img_shape)
        except Exception:
            pass

    return None


# ---- Geometry helpers ----
def _angle_abc(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Optional[float]:
    """Returns angle ABC in degrees (at point B)."""
    if a is None or b is None or c is None:
        return None
    bax = a[0] - b[0]; bay = a[1] - b[1]
    bcx = c[0] - b[0]; bcy = c[1] - b[1]
    na = math.hypot(bax, bay); nc = math.hypot(bcx, bcy)
    if na == 0 or nc == 0:
        return None
    cosang = max(-1.0, min(1.0, (bax*bcx + bay*bcy) / (na*nc)))
    return math.degrees(math.acos(cosang))

def _angle_to_ref(vx: float, vy: float, rx: float, ry: float) -> Optional[float]:
    """Angle (0–180) between vector v and reference r."""
    norm_v = math.hypot(vx, vy)
    norm_r = math.hypot(rx, ry)
    if norm_v == 0 or norm_r == 0:
        return None
    cosang = max(-1.0, min(1.0, (vx*rx + vy*ry) / (norm_v*norm_r)))
    return math.degrees(math.acos(cosang))


# ---- Main metric function ----
def calculate_metrics(keypoints: Any,
                      img_shape: Optional[Tuple[int, int, int]] = None,
                      handedness: str = "right") -> Dict[str, Optional[float]]:
    """
    Compute the 4 required per-frame metrics:
      - elbow_angle: front shoulder–elbow–wrist angle (deg)
      - spine_lean: angle of hip→shoulder midline vs vertical up (deg)
      - head_knee: |x_head - x_front_knee| (pixels)
      - foot_dir: angle of ankle→toe vs x-axis (deg)
    Inputs:
      keypoints: flexible structure (dict, list, or list of triplets). Values can be pixels or normalized [0..1].
      img_shape: (H, W, C) if you want normalized coords scaled to pixels automatically.
      handedness: "right" (front = LEFT) or "left" (front = RIGHT).
    """
    # Core landmarks (both sides)
    L_SHO = _extract_point(keypoints, "LEFT_SHOULDER", img_shape)
    L_ELB = _extract_point(keypoints, "LEFT_ELBOW", img_shape)
    L_WRI = _extract_point(keypoints, "LEFT_WRIST", img_shape)
    L_HIP = _extract_point(keypoints, "LEFT_HIP", img_shape)
    L_KNE = _extract_point(keypoints, "LEFT_KNEE", img_shape)
    L_ANK = _extract_point(keypoints, "LEFT_ANKLE", img_shape)
    L_TOE = _extract_point(keypoints, "LEFT_FOOT_INDEX", img_shape)
    L_EAR = _extract_point(keypoints, "LEFT_EAR", img_shape)

    R_SHO = _extract_point(keypoints, "RIGHT_SHOULDER", img_shape)
    R_ELB = _extract_point(keypoints, "RIGHT_ELBOW", img_shape)
    R_WRI = _extract_point(keypoints, "RIGHT_WRIST", img_shape)
    R_HIP = _extract_point(keypoints, "RIGHT_HIP", img_shape)
    R_KNE = _extract_point(keypoints, "RIGHT_KNEE", img_shape)
    R_ANK = _extract_point(keypoints, "RIGHT_ANKLE", img_shape)
    R_TOE = _extract_point(keypoints, "RIGHT_FOOT_INDEX", img_shape)
    R_EAR = _extract_point(keypoints, "RIGHT_EAR", img_shape)

    # Midpoints for spine (hip→shoulder)
    SHO_M = None
    HIP_M = None
    if L_SHO and R_SHO:
        SHO_M = ((L_SHO[0] + R_SHO[0]) / 2.0, (L_SHO[1] + R_SHO[1]) / 2.0)
    if L_HIP and R_HIP:
        HIP_M = ((L_HIP[0] + R_HIP[0]) / 2.0, (L_HIP[1] + R_HIP[1]) / 2.0)

    # Determine "front" side from handedness (right-handed batter → LEFT is front)
    front_is_left = (handedness.lower() == "right")
    if front_is_left:
        SHO_F, ELB_F, WRI_F, KNE_F, ANK_F, TOE_F, EAR_F = L_SHO, L_ELB, L_WRI, L_KNE, L_ANK, L_TOE, L_EAR
    else:
        SHO_F, ELB_F, WRI_F, KNE_F, ANK_F, TOE_F, EAR_F = R_SHO, R_ELB, R_WRI, R_KNE, R_ANK, R_TOE, R_EAR

    # 1) Front elbow angle (deg)
    elbow_angle = _angle_abc(SHO_F, ELB_F, WRI_F)

    # 2) Spine lean (deg) = angle between (HIP_M→SHO_M) and vertical up (0, -1)
    spine_lean = None
    if SHO_M is not None and HIP_M is not None:
        vx = SHO_M[0] - HIP_M[0]
        vy = SHO_M[1] - HIP_M[1]
        spine_lean = _angle_to_ref(vx, vy, 0.0, -1.0)  # 0° = perfectly vertical

    # 3) Head-over-knee horizontal offset (pixels) — use ear as head proxy
    head_knee = None
    if EAR_F is not None and KNE_F is not None:
        head_knee = abs(EAR_F[0] - KNE_F[0])

    # 4) Front foot direction (deg) — ankle→toe vs x-axis (1,0)
    foot_dir = None
    if ANK_F is not None and TOE_F is not None:
        vx = TOE_F[0] - ANK_F[0]
        vy = TOE_F[1] - ANK_F[1]
        foot_dir = _angle_to_ref(vx, vy, 1.0, 0.0)  # 0° = pointing exactly along +x

    # Round and return
    def rnd(x, n=1):
        return None if x is None else (round(float(x), n))

    # head_knee can be large; keep as int pixels if available
    return {
        "elbow_angle": rnd(elbow_angle, 1),
        "spine_lean": rnd(spine_lean, 1),
        "head_knee": None if head_knee is None else int(round(head_knee)),
        "foot_dir": rnd(foot_dir, 1),
    }
