"""
Extracts motor performance test data from a table image using EasyOCR,
then saves the structured results to a CSV file.

Works on:
  - Simple single-motor tables (motor_table.jpg)
  - Full spec-sheet images with multiple KV-variant test tables (motor_table_2.jpg)

Usage:
    python parse_motor_image.py [image_path] [--output path]

    image_path defaults to data/images/motor_table.jpg
"""

import argparse
import re
import sys
import types
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# python-bidi fails to build on Python 3.14; stub it out (only needed for RTL scripts).
_bidi = types.ModuleType("bidi")
_bidi.get_display = lambda text, *a, **kw: text  # type: ignore[attr-defined]
sys.modules.setdefault("bidi", _bidi)
sys.modules.setdefault("bidi.algorithm", _bidi)

import easyocr  # noqa: E402 — must come after bidi stub

DEFAULT_IMAGE = Path("data/images/motor_table.jpg")
OUTPUT_CSV = Path("data/motor_test_points.csv")

# Sanity ranges for each numeric column
ROW_RANGES = {
    "throttle_pct":       (35,   105),
    "voltage_v":          (10.0,  35.0),
    "current_a":          (0.5,  200.0),
    "thrust_g":           (50,   8000),
    "rpm":                (1000, 50000),
    "power_w":            (5,    8000),
    "efficiency_g_per_w": (0.3,  20.0),
}

# Prop label patterns — covers common naming conventions.
# Use [x*] to match both "x" and "*" separators (OCR often reads × as *).
# H[QO] handles the common OCR confusion of Q → O.
PROP_PATTERNS = [
    r"GF\s*\d{4}",                          # GF7035, GF8040
    r"\d+[x*]\d+(?:[.,]\d+)?[x*]\d+",      # 5x4.5x3 or 5*4.5*3
    r"H[QO]\s*\d[\d.]*",                    # HQ7.0 or HO7.0 (OCR Q→O)
    r"T\d+[x*]\d+",                         # T-motor style T5045
    r"DAL\s*\d",                            # DAL props
    r"SL\s*\d{4}",                          # SL4525, SL5125 (Sunnysky-style)
]

# Column header keywords → field names.  Used to detect the header row and
# derive column x-positions for position-aware numeric assignment.
HEADER_KEYWORDS = {
    "throttle":   "throttle_pct",
    "voltage":    "voltage_v",
    "current":    "current_a",
    "thrust":     "thrust_g",
    "rpm":        "rpm",
    "power":      "power_w",
    "efficiency": "efficiency_g_per_w",
    "thrust/w":   "efficiency_g_per_w",
    "g/w":        "efficiency_g_per_w",
}


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def detect_table_bboxes(img_gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Use morphological line detection to find table regions in a grayscale image.
    Returns a list of (x1, y1, x2, y2) bboxes sorted top-to-bottom.
    Falls back to the full image if no table borders are detected.
    """
    h, w = img_gray.shape
    binary = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    # Horizontal lines: kernel width ≥ 10% of image width
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 10), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    # Vertical lines: kernel height ≥ 3% of image height
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 30)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    grid = cv2.add(h_lines, v_lines)
    # Dilate to bridge nearby line segments into solid table blocks.
    # iterations=1 preserves gaps between adjacent tables that are only a few
    # pixels apart (more iterations merges them into one large region).
    grid = cv2.dilate(grid, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = h * w * 0.03  # ignore regions smaller than 3% of image
    rects = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        pad_x = int(bw * 0.05)
        pad_top = int(bh * 0.05)
        pad_bot = int(bh * 0.20)  # tables often lack a bottom border; over-extend down
        rects.append((
            max(0, x - pad_x), max(0, y - pad_top),
            min(w, x + bw + pad_x), min(h, y + bh + pad_bot),
        ))
    rects.sort(key=lambda r: r[1])
    return rects if rects else [(0, 0, w, h)]


def preprocess_for_ocr(img_gray_crop: np.ndarray) -> np.ndarray:
    """
    Preprocess a cropped table region for OCR.

    Scales up small crops, then applies contrast/sharpness boost (same approach
    that worked on motor_table.jpg).  For very low-contrast crops, falls back to
    adaptive binarization to recover faint text.
    """
    from PIL import ImageEnhance as _IE
    crop_h = img_gray_crop.shape[0]
    if crop_h < 600:
        scale = 600 / crop_h
        img_gray_crop = cv2.resize(
            img_gray_crop,
            (int(img_gray_crop.shape[1] * scale), 600),
            interpolation=cv2.INTER_CUBIC,
        )
    pil = Image.fromarray(img_gray_crop)
    pil = _IE.Contrast(pil).enhance(1.5)
    pil = _IE.Sharpness(pil).enhance(2.0)
    enhanced = np.array(pil)

    # If the crop is very low contrast (compressed JPEG / photo background),
    # adaptive binarization recovers more text than contrast boosting.
    contrast_range = int(enhanced.max()) - int(enhanced.min())
    if contrast_range < 80:
        denoised = cv2.fastNlMeansDenoising(img_gray_crop, h=10)
        return cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )
    return enhanced


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def looks_like_prop(text: str) -> bool:
    if re.search(r"\dKV", text, re.IGNORECASE):  # motor spec, not a prop
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in PROP_PATTERNS)


def group_into_rows(detections: list[dict], y_tol: int = 10) -> list[list[dict]]:
    """Cluster OCR detections by similar y-centre into rows, sorted left→right."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["cy"])
    rows, current = [], [detections[0]]
    for det in detections[1:]:
        if abs(det["cy"] - current[-1]["cy"]) <= y_tol:
            current.append(det)
        else:
            rows.append(sorted(current, key=lambda d: d["cx"]))
            current = [det]
    rows.append(sorted(current, key=lambda d: d["cx"]))
    return rows


def to_float(text: str) -> Optional[float]:
    t = text.strip().rstrip("%").rstrip("_")  # strip trailing underscore OCR artifact
    t = t.replace(",", ".")
    t = re.sub(r"\.{2,}", ".", t)                           # "55,.20" → "55.20"
    t = re.sub(r"^(\d+)_(\d{1,2})$", r"\1.\2", t)          # "235_20" → "235.20"
    t = re.sub(r"^(\d+\.)\s+(\d+)$", r"\1\2", t)           # "103. 20" → "103.20"
    t = re.sub(r"^(\d+(?:\.\d+)?)\s+[^\d].*$", r"\1", t)   # "60 QQ" → "60"
    t = re.sub(r"^(\d+)\s+(\d{1,3})$", r"\1.\2", t)        # "23 8" → "23.8"
    try:
        return float(t)
    except ValueError:
        return None


def in_range(val: float, key: str) -> bool:
    lo, hi = ROW_RANGES[key]
    return lo <= val <= hi


def try_assign_columns(nums: list[float]) -> Optional[dict]:
    """
    Given a list starting with [Voltage, Current, ...], try all known column
    orderings and return parsed values or None.

    Order A: V, A, Thrust, RPM,   Power, Eff  (motor_table.jpg)
    Order B: V, A, RPM,   Thrust, Power, Eff  (motor_table_2.jpg)
    Order C: V, A, Thrust, Power, Eff,   RPM  (motor_table_3.webp — Power before RPM)
             Validated by P ≈ V×A since there is no fifth-position power to check.
    """
    if len(nums) < 4:
        return None
    v, a = nums[0], nums[1]
    if not (in_range(v, "voltage_v") and in_range(a, "current_a")):
        return None

    n2, n3 = nums[2], nums[3]

    def _eff(idx: int) -> Optional[float]:
        return nums[idx] if len(nums) > idx and in_range(nums[idx], "efficiency_g_per_w") else None

    # Orders A and B require Power at index 4
    if len(nums) >= 5:
        pw = nums[4]
        if in_range(pw, "power_w"):
            eff = _eff(5)
            # Order A: Thrust, RPM
            if in_range(n2, "thrust_g") and in_range(n3, "rpm"):
                return {"voltage_v": round(v, 2), "current_a": round(a, 2),
                        "thrust_g": int(n2), "rpm": int(n3),
                        "power_w": round(pw, 2),
                        "efficiency_g_per_w": round(eff, 2) if eff is not None else None}
            # Order B: RPM, Thrust
            if in_range(n2, "rpm") and in_range(n3, "thrust_g"):
                return {"voltage_v": round(v, 2), "current_a": round(a, 2),
                        "thrust_g": int(n3), "rpm": int(n2),
                        "power_w": round(pw, 2),
                        "efficiency_g_per_w": round(eff, 2) if eff is not None else None}

    # Order C: Thrust at n2, Power at n3; validate P ≈ V×A; RPM is optional/later
    if in_range(n2, "thrust_g") and in_range(n3, "power_w"):
        expected = v * a
        if abs(n3 - expected) / max(expected, 1) < 0.25:
            c_eff = _eff(4)
            c_rpm = next((int(x) for x in nums[5:] if in_range(x, "rpm")), None)
            return {"voltage_v": round(v, 2), "current_a": round(a, 2),
                    "thrust_g": int(n2), "rpm": c_rpm,
                    "power_w": round(n3, 2),
                    "efficiency_g_per_w": round(c_eff, 2) if c_eff is not None else None}

        # Decimal-dropped current: OCR reads "4.3" as "43".  Try a/10 if that
        # brings P ≈ V × (a/10) within tolerance (Order C layout only).
        a_tenth = a / 10
        if in_range(a_tenth, "current_a"):
            expected10 = v * a_tenth
            if abs(n3 - expected10) / max(expected10, 1) < 0.25:
                c_eff = _eff(4)
                c_rpm = next((int(x) for x in nums[5:] if in_range(x, "rpm")), None)
                return {"voltage_v": round(v, 2), "current_a": round(a_tenth, 2),
                        "thrust_g": int(n2), "rpm": c_rpm,
                        "power_w": round(n3, 2),
                        "efficiency_g_per_w": round(c_eff, 2) if c_eff is not None else None}

    return None


def try_parse_data_row(texts: list[str]) -> Optional[dict]:
    """
    Extract numeric columns from one OCR row.

    Attempt 1 — throttle present: scan for a throttle value (35–105, mult of 5)
    then parse the remaining columns.

    Attempt 2 — throttle absent (OCR missed it): scan starting from any voltage-
    range value and validate with Power ≈ V × A (within 25%).
    """
    nums = [v for t in texts if (v := to_float(t)) is not None]

    # Attempt 1: throttle explicitly detected
    for i, n in enumerate(nums):
        if not in_range(n, "throttle_pct") or n % 5 != 0:
            continue
        result = try_assign_columns(nums[i + 1:])
        if result:
            return {"throttle_pct": int(n), **result}

    # Attempt 2: no throttle — look for voltage-starting sequence
    for start in range(len(nums)):
        if not in_range(nums[start], "voltage_v"):
            continue
        result = try_assign_columns(nums[start:])
        if result:
            expected_pw = result["voltage_v"] * result["current_a"]
            if abs(result["power_w"] - expected_pw) / max(expected_pw, 1) < 0.25:
                return {"throttle_pct": None, **result}

    return None


def detect_column_positions(rows: list[list[dict]]) -> Optional[dict[str, float]]:
    """
    Scan grouped rows for a header row (≥4 recognised column keywords).
    Returns {field_name: x_center} or None if no header row is found.
    """
    for row in rows:
        matches: dict[str, float] = {}
        for d in row:
            t = d["text"].lower()
            for kw, field in HEADER_KEYWORDS.items():
                if kw in t:
                    matches[field] = d["cx"]
        if len(matches) >= 4:
            return matches
    return None


def try_parse_data_row_with_cols(
    row: list[dict],
    col_positions: dict[str, float],
) -> Optional[dict]:
    """
    Position-aware row parser.  Each numeric token is assigned to the nearest
    column by x-distance; range validation then confirms the assignment.
    Returns None if required fields are missing or Power ≈ V×A check fails.
    """
    REQUIRED = {"voltage_v", "current_a", "power_w"}
    assigned: dict[str, Optional[float]] = {f: None for f in ROW_RANGES}

    for d in row:
        val = to_float(d["text"])
        if val is None:
            continue
        best_field = min(col_positions, key=lambda f: abs(col_positions[f] - d["cx"]))
        if in_range(val, best_field) and assigned[best_field] is None:
            assigned[best_field] = val

    if any(assigned[f] is None for f in REQUIRED):
        return None

    v = assigned["voltage_v"]
    a = assigned["current_a"]
    pw = assigned["power_w"]
    expected_pw = v * a  # type: ignore[operator]
    if abs(pw - expected_pw) / max(expected_pw, 1) > 0.25:  # type: ignore[operator]
        return None

    def _int(x: Optional[float]) -> Optional[int]:
        return int(x) if x is not None else None

    def _rnd(x: Optional[float]) -> Optional[float]:
        return round(x, 2) if x is not None else None

    return {
        "throttle_pct":       _int(assigned["throttle_pct"]),
        "voltage_v":          _rnd(v),
        "current_a":          _rnd(a),
        "thrust_g":           _int(assigned["thrust_g"]),
        "rpm":                _int(assigned["rpm"]),
        "power_w":            _rnd(pw),
        "efficiency_g_per_w": _rnd(assigned["efficiency_g_per_w"]),
    }


def get_label_for_y(
    cy: float,
    labels: list[tuple[float, str]],
    row_height: float,
) -> Optional[str]:
    """
    For center-anchored labels (prop merged cells): the boundary between
    section i and i+1 is their midpoint shifted forward by 0.6 row heights.
    """
    if not labels:
        return None
    if len(labels) == 1:
        return labels[0][1]
    for i in range(len(labels) - 1):
        boundary = (labels[i][0] + labels[i + 1][0]) / 2 + row_height * 0.6
        if cy < boundary:
            return labels[i][1]
    return labels[-1][1]


def get_motor_for_y(
    cy: float,
    motor_sections: list[tuple[float, str]],
) -> Optional[str]:
    """
    For top-anchored labels (test-report headers): a section begins at its
    header y-position and ends just before the next header.
    """
    if not motor_sections:
        return None
    result = motor_sections[0][1]
    for y, name in motor_sections:
        if cy >= y:
            result = name
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def run(image_path: Path) -> pd.DataFrame:
    if not image_path.exists():
        sys.exit(
            f"Image not found: {image_path}\n"
            "Place your image there and re-run, or pass a path as the first argument."
        )

    print(f"Loading image: {image_path}")
    img_pil = Image.open(image_path).convert("RGB")
    w, h = img_pil.size
    print(f"  Image size: {w}x{h}px")

    img_gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)

    # Detect table bounding boxes via morphological line detection
    bboxes = detect_table_bboxes(img_gray)
    print(f"  Table regions: {len(bboxes)} detected  "
          f"({['full image fallback' if bboxes == [(0,0,w,h)] else str(bboxes)][0]})")

    print("Running OCR (first run downloads ~40 MB model)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # OCR each table region separately for better per-region accuracy
    # (full-image OCR degrades small decimal recognition on tall images).
    # Bboxes may overlap due to padding; deduplicate tokens afterward.
    raw_detections: list[dict] = []
    for x1, y1, x2, y2 in bboxes:
        crop = preprocess_for_ocr(img_gray[y1:y2, x1:x2])
        crop_h, crop_w = crop.shape[:2]
        results = reader.readtext(crop, paragraph=False, canvas_size=max(crop_w, crop_h))
        sx = (x2 - x1) / crop_w
        sy = (y2 - y1) / crop_h
        for bbox_pts, text, conf in results:
            if conf < 0.3 or not text.strip():
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            raw_detections.append({
                "text": text.strip(),
                "cx":   x1 + (sum(xs) / 4) * sx,
                "cy":   y1 + (sum(ys) / 4) * sy,
                "conf": conf,
            })

    # Deduplicate: when two detections are within 15px in both axes (overlap
    # zone), keep the one with higher confidence.
    raw_detections.sort(key=lambda d: -d["conf"])  # process highest-conf first
    suppressed: set[int] = set()
    for i, d in enumerate(raw_detections):
        if i in suppressed:
            continue
        for j in range(i + 1, len(raw_detections)):
            if j in suppressed:
                continue
            e = raw_detections[j]
            if abs(d["cx"] - e["cx"]) < 15 and abs(d["cy"] - e["cy"]) < 15:
                suppressed.add(j)  # d has higher conf; suppress e

    all_detections = [
        {"text": d["text"], "cx": d["cx"], "cy": d["cy"]}
        for i, d in enumerate(raw_detections)
        if i not in suppressed
    ]

    # Estimate row height before grouping so we can scale y_tol
    throttle_ys = sorted(
        d["cy"] for d in all_detections if re.match(r"^\d+%$", d["text"])
    )
    row_height = 40.0
    if len(throttle_ys) >= 2:
        row_height = float(np.median(np.diff(throttle_ys)))

    y_tol = max(8, int(row_height * 0.2))
    rows = group_into_rows(all_detections, y_tol=y_tol)
    print(f"  Detected {len(all_detections)} text regions -> {len(rows)} rows  "
          f"(row_height~{row_height:.0f}px, y_tol={y_tol}px)")

    # Detect column positions from header row (enables position-aware parsing)
    col_positions = detect_column_positions(rows)
    if col_positions:
        print(f"  Column header found: {list(col_positions.keys())}")

    # ------------------------------------------------------------------
    # PASS 1 — find motor sections and prop sections
    # ------------------------------------------------------------------
    af_part: Optional[str] = None
    kv_part: Optional[str] = None
    motor_sections: list[tuple[float, str]] = []  # (cy, motor_type_string)
    kv_fallbacks: list[tuple[float, str]] = []    # (cy, "NNNNkv") — lower confidence
    prop_labels_raw: list[tuple[float, str]] = []  # (cy, raw_text)

    for d in all_detections:
        text = d["text"]
        cx, cy = d["cx"], d["cy"]
        fixed = re.sub(r"[Oo]", "0", text)  # fix O→0 OCR error

        # --- Motor type: left-column "AF310 / KV1210" style ---
        if cx < 200:
            if re.match(r"AF\s*\d+", fixed, re.IGNORECASE) and "KV" not in fixed.upper():
                af_part = re.sub(r"\s+", "", fixed).upper()
            elif re.match(r"KV\s*\d+", fixed, re.IGNORECASE):
                kv_part = re.sub(r"\s+", "", fixed).upper()

        # --- Motor type: "Avenger 2806.5-1300KV Test Report" style header ---
        if re.search(r"test\s+report", text, re.IGNORECASE):
            m = re.search(
                r"([\w.\-]+\s+\d+(?:\.\d+)?(?:\s*-\s*\d+)?KV)",
                text, re.IGNORECASE
            )
            if m:
                name = re.sub(r"\s*-\s*", "-", m.group(1).strip())
                motor_sections.append((cy, name))

        # --- Motor type fallback: bare "1300KV" or embedded "ECOII2004-1600KV" style ---
        m_kv = re.fullmatch(r"(\d{3,4})\s*[Kk][Vv]", text.strip())
        if m_kv:
            kv_fallbacks.append((cy, m_kv.group(0).upper()))
        elif re.search(r"\d{3,4}\s*[Kk][Vv]", text) and not re.search(r"test\s+report", text, re.IGNORECASE):
            # e.g. "ECOII2004-1600KV" — grab product name + KV as the motor label
            m_full = re.search(r"([A-Za-z]\w*[-.]?\d+[-.]?\d*\s*-?\s*\d{3,4}\s*[Kk][Vv])", text)
            if m_full:
                kv_fallbacks.append((cy, m_full.group(1).strip()))

        # --- Prop labels ---
        if looks_like_prop(text):
            prop_labels_raw.append((cy, text.strip()))

    # Build left-column motor type (Pattern 1)
    lc_motor_type: Optional[str] = None
    if af_part and kv_part:
        lc_motor_type = f"{af_part} {kv_part}"
    elif af_part or kv_part:
        lc_motor_type = af_part or kv_part

    # Merge nearby "3-blades" / "2-blades" suffix tokens into the preceding prop label
    suffix_pat = re.compile(r"\d-blades?", re.IGNORECASE)
    prop_labels: list[tuple[float, str]] = []
    for cy, name in sorted(prop_labels_raw, key=lambda x: x[0]):
        if suffix_pat.search(name):
            if prop_labels and abs(cy - prop_labels[-1][0]) < row_height * 2:
                prev_cy, prev_name = prop_labels.pop()
                prop_labels.append((prev_cy, f"{prev_name} {name}".strip()))
                continue
        prop_labels.append((cy, name))

    # Merge KV fallbacks: for any KV value not already covered by a full motor-section
    covered_kvs = {re.search(r"\d{3,4}KV", ms, re.IGNORECASE).group(0).upper()
                   for _, ms in motor_sections
                   if re.search(r"\d{3,4}KV", ms, re.IGNORECASE)}
    for cy, kv_str in sorted(kv_fallbacks, key=lambda x: x[0]):
        if kv_str not in covered_kvs:
            motor_sections.append((cy, kv_str))
            covered_kvs.add(kv_str)

    motor_sections.sort(key=lambda x: x[0])
    prop_labels.sort(key=lambda x: x[0])

    print(f"  Motor sections : {[(m, f'y={y:.0f}') for y, m in motor_sections] or lc_motor_type or 'Unknown'}")
    print(f"  Prop sections  : {[(p, f'y={y:.0f}') for y, p in prop_labels]}")

    # ------------------------------------------------------------------
    # PASS 2 — parse data rows
    # ------------------------------------------------------------------
    data = []
    for row in rows:
        texts = [d["text"] for d in row]
        cy = row[0]["cy"]

        # Assign motor type (top-anchored: section starts at header y)
        if motor_sections:
            motor_type = get_motor_for_y(cy, motor_sections)
        else:
            motor_type = lc_motor_type or "Unknown"

        # Assign prop section
        props = get_label_for_y(cy, prop_labels, row_height) if prop_labels else None

        # Use position-aware parser when the header covers ≥6 distinct columns.
        # Always fall back to the positional parser if it returns nothing (e.g.
        # because a required column like voltage_v was not in the header).
        if col_positions and len(col_positions) >= 6:
            row_data = try_parse_data_row_with_cols(row, col_positions)
            if row_data is None:
                row_data = try_parse_data_row(texts)
        else:
            row_data = try_parse_data_row(texts)

        if row_data and props:
            row_data["motor_type"] = motor_type
            row_data["props"] = props
            row_data["_image_y"] = cy
            data.append(row_data)

    if not data:
        print("\n--- DEBUG: rows with 3+ numeric tokens (likely data rows) ---")
        for row in rows:
            texts = [d["text"] for d in row]
            nums = [v for t in texts if (v := to_float(t)) is not None]
            if len(nums) >= 3:
                cy = row[0]["cy"]
                print(f"  y={cy:.0f}  texts={texts}  nums={nums}")
        sys.exit(
            "\nNo data rows found. Check that the image is a clear photo/scan of the "
            "motor performance table and that OCR can read the text."
        )

    cols = ["motor_type", "props", "throttle_pct", "voltage_v", "current_a",
            "thrust_g", "rpm", "power_w", "efficiency_g_per_w"]
    df = pd.DataFrame(data).sort_values("_image_y").reset_index(drop=True)[cols]
    return df


def main():
    parser = argparse.ArgumentParser(description="Parse motor performance image to CSV")
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE,
                        help=f"Path to the image file (default: {DEFAULT_IMAGE})")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    df = run(args.image)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
