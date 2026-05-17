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
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def looks_like_prop(text: str) -> bool:
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
    t = text.strip().rstrip("%").replace(",", ".")
    # Fix OCR error: decimal point read as space (e.g. "23 8" → "23.8")
    t = re.sub(r"^(\d+)\s+(\d{1,3})$", r"\1.\2", t)
    try:
        return float(t)
    except ValueError:
        return None


def in_range(val: float, key: str) -> bool:
    lo, hi = ROW_RANGES[key]
    return lo <= val <= hi


def try_assign_columns(nums: list[float]) -> Optional[dict]:
    """
    Given a list starting with [Voltage, Current, ...], try both known column
    orderings and return parsed values or None.

    Order A (motor_table.jpg):   V, A, Thrust, RPM, Power, Eff
    Order B (motor_table_2.jpg): V, A, RPM,   Thrust, Power, Eff
    """
    if len(nums) < 5:
        return None
    v, a = nums[0], nums[1]
    if not (in_range(v, "voltage_v") and in_range(a, "current_a")):
        return None

    n2, n3, pw = nums[2], nums[3], nums[4]
    if not in_range(pw, "power_w"):
        return None

    # Efficiency is optional — ignore if out of range (could be temperature)
    eff = nums[5] if len(nums) > 5 and in_range(nums[5], "efficiency_g_per_w") else None

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
    header y-position and ends just before the next header.  Walk forward
    through sorted labels and keep updating the result as long as cy >= label_y.
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
    from PIL import ImageEnhance
    img_pil = Image.open(image_path).convert("RGB")
    w, h = img_pil.size
    print(f"  Image size: {w}x{h}px")

    # Boost contrast and sharpness before OCR — helps with faint/compressed table text
    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.5)
    img_pil = ImageEnhance.Sharpness(img_pil).enhance(2.0)
    img = np.array(img_pil)

    print("Running OCR (first run downloads ~40 MB model)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    # canvas_size must match or exceed the image's longest dimension to prevent
    # EasyOCR from downscaling large spec sheets before detection.
    results = reader.readtext(img, paragraph=False, canvas_size=max(w, h))

    # Build detection list
    detections = []
    for bbox, text, conf in results:
        if conf < 0.1 or not text.strip():
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        detections.append({
            "text": text.strip(),
            "cx":   sum(xs) / 4,
            "cy":   sum(ys) / 4,
        })

    # Estimate row height before grouping so we can scale y_tol
    throttle_ys = sorted(
        d["cy"] for d in detections if re.match(r"^\d+%$", d["text"])
    )
    row_height = 40.0
    if len(throttle_ys) >= 2:
        row_height = float(np.median(np.diff(throttle_ys)))

    y_tol = max(8, int(row_height * 0.2))
    rows = group_into_rows(detections, y_tol=y_tol)
    print(f"  Detected {len(results)} text regions → {len(rows)} rows  "
          f"(row_height≈{row_height:.0f}px, y_tol={y_tol}px)")

    # ------------------------------------------------------------------
    # PASS 1 — find motor sections and prop sections
    # ------------------------------------------------------------------
    af_part: Optional[str] = None
    kv_part: Optional[str] = None
    motor_sections: list[tuple[float, str]] = []  # (cy, motor_type_string)
    kv_fallbacks: list[tuple[float, str]] = []    # (cy, "NNNNkv") — lower confidence
    prop_labels_raw: list[tuple[float, str]] = []  # (cy, raw_text)

    for d in detections:
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
                r"([\w.\-]+\s+\d+(?:\.\d+)?(?:-\d+)?KV)",
                text, re.IGNORECASE
            )
            if m:
                motor_sections.append((cy, m.group(1).strip()))

        # --- Motor type fallback: bare "1300KV" / "1700kv" token anywhere ---
        # Used when the full "Test Report" header line isn't captured by OCR.
        m_kv = re.fullmatch(r"(\d{3,4})\s*[Kk][Vv]", text.strip())
        if m_kv:
            kv_fallbacks.append((cy, m_kv.group(0).upper()))

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
            # Attach suffix to the most recent prop label if within 2 row heights
            if prop_labels and abs(cy - prop_labels[-1][0]) < row_height * 2:
                prev_cy, prev_name = prop_labels.pop()
                merged_name = f"{prev_name} {name}".strip()
                prop_labels.append((prev_cy, merged_name))
                continue
        prop_labels.append((cy, name))

    # Merge KV fallbacks: for any KV value not already covered by a full motor-section
    # name, add it so that section boundaries are still computed correctly.
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

        row_data = try_parse_data_row(texts)
        if row_data and props:
            row_data["motor_type"] = motor_type
            row_data["props"] = props
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
    df = pd.DataFrame(data)[cols]
    # throttle_pct and rpm are nullable; sort NaN throttle rows last within each group
    df = df.sort_values(
        ["motor_type", "props", "throttle_pct"],
        na_position="last",
    ).reset_index(drop=True)
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
