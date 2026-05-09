"""
Extracts motor performance test data from a table image using EasyOCR,
then saves the structured results to a CSV file.

Usage:
    python parse_motor_image.py [image_path]

    image_path defaults to data/images/motor_table.jpg
"""

import argparse
import re
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# python-bidi fails to build on Python 3.14 but is only used for
# right-to-left scripts (Arabic/Hebrew) — stub it out for English-only use.
_bidi = types.ModuleType("bidi")
_bidi.get_display = lambda text, *a, **kw: text  # type: ignore[attr-defined]
sys.modules.setdefault("bidi", _bidi)
sys.modules.setdefault("bidi.algorithm", _bidi)

import easyocr  # noqa: E402 — must come after the bidi stub

DEFAULT_IMAGE = Path("data/images/motor_table.jpg")
OUTPUT_CSV = Path("data/motor_test_points.csv")

# Expected value ranges for sanity-checking detected rows
ROW_RANGES = {
    "throttle_pct": (35, 105),
    "voltage_v":    (20.0, 30.0),
    "current_a":    (0.5, 120.0),
    "thrust_g":     (100, 5000),
    "rpm":          (5000, 35000),
    "power_w":      (10, 5000),
    "efficiency_g_per_w": (0.5, 15.0),
}


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
    """Try to parse text as a float; return None on failure."""
    try:
        return float(text.strip().rstrip("%").replace(",", "."))
    except ValueError:
        return None


def in_range(val: float, key: str) -> bool:
    lo, hi = ROW_RANGES[key]
    return lo <= val <= hi


def try_parse_data_row(texts: list[str]) -> Optional[dict]:
    """
    Given the text tokens from one table row, extract the 7 numeric columns.
    Returns a dict with numeric fields, or None if the row doesn't match.
    """
    nums = [v for t in texts if (v := to_float(t)) is not None]

    # Scan for a throttle value (integer multiple of 5 between 35–105)
    for i, n in enumerate(nums):
        if not in_range(n, "throttle_pct"):
            continue
        if n % 5 != 0:
            continue
        rest = nums[i + 1 :]
        if len(rest) < 6:
            continue
        v, a, thr, rpm, pw, eff = rest[:6]
        if (
            in_range(v,   "voltage_v") and
            in_range(a,   "current_a") and
            in_range(thr, "thrust_g") and
            in_range(rpm, "rpm") and
            in_range(pw,  "power_w") and
            in_range(eff, "efficiency_g_per_w")
        ):
            return {
                "throttle_pct": int(n),
                "voltage_v":    round(v,   2),
                "current_a":    round(a,   2),
                "thrust_g":     int(thr),
                "rpm":          int(rpm),
                "power_w":      round(pw,  2),
                "efficiency_g_per_w": round(eff, 2),
            }
    return None


def run(image_path: Path) -> pd.DataFrame:
    if not image_path.exists():
        sys.exit(
            f"Image not found: {image_path}\n"
            "Place your image there and re-run, or pass a path as the first argument."
        )

    print(f"Loading image: {image_path}")
    img = np.array(Image.open(image_path).convert("RGB"))

    print("Running OCR (first run downloads ~40 MB model)...")
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    results = reader.readtext(img)

    # Build detection list with centre coordinates
    detections = []
    for bbox, text, conf in results:
        if conf < 0.2 or not text.strip():
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        detections.append({
            "text": text.strip(),
            "cx":   sum(xs) / 4,
            "cy":   sum(ys) / 4,
        })

    rows = group_into_rows(detections, y_tol=10)
    print(f"  Detected {len(results)} text regions → {len(rows)} rows")

    # PASS 1: scan all detections to find motor type, prop-label y-positions,
    # and average row height (used to offset the section boundary).
    motor_type: Optional[str] = None
    label_y: dict[str, float] = {}
    af_part: Optional[str] = None
    kv_part: Optional[str] = None

    throttle_ys: list[float] = []

    for d in detections:
        text = d["text"]
        cx, cy = d["cx"], d["cy"]

        # Motor type sits in the leftmost column (cx < 150).
        # OCR often reads "0" as "O" — normalise before matching.
        if cx < 150:
            fixed = re.sub(r"[Oo]", "0", text)
            if re.match(r"AF\s*\d+", fixed, re.IGNORECASE):
                af_part = re.sub(r"\s+", "", fixed).upper()
            elif re.match(r"KV\s*\d+", fixed, re.IGNORECASE):
                kv_part = re.sub(r"\s+", "", fixed).upper()

        # Prop section labels
        if re.search(r"GF\s*7035", text, re.IGNORECASE):
            label_y["GF7035 3-blades"] = cy
        if re.search(r"GF\s*8040", text, re.IGNORECASE):
            label_y["GF8040 3-blades"] = cy

        # Collect throttle row y-positions to estimate row height
        if re.match(r"^\d+%$", text):
            throttle_ys.append(cy)

    if af_part and kv_part:
        motor_type = f"{af_part} {kv_part}"
    elif af_part or kv_part:
        motor_type = af_part or kv_part

    # Estimate row height from median spacing of throttle detections
    row_height = 40.0
    if len(throttle_ys) >= 2:
        row_height = float(np.median(np.diff(sorted(throttle_ys))))

    # Section boundary: midpoint between the two prop labels, shifted forward by
    # half a row so the last row of the first section isn't misclassified.
    y_boundary: Optional[float] = None
    if "GF7035 3-blades" in label_y and "GF8040 3-blades" in label_y:
        y_boundary = (label_y["GF7035 3-blades"] + label_y["GF8040 3-blades"]) / 2 + row_height * 0.6
        print(f"  Motor type: {motor_type}")
        print(f"  Row height: {row_height:.1f}px  "
              f"Section boundary: y={y_boundary:.0f}px  "
              f"(GF7035@{label_y['GF7035 3-blades']:.0f}, "
              f"GF8040@{label_y['GF8040 3-blades']:.0f})")

    # PASS 2: parse data rows, assigning props by y-position vs boundary
    current_props: Optional[str] = None
    data = []

    for row in rows:
        texts = [d["text"] for d in row]
        cy = row[0]["cy"]

        if y_boundary is not None:
            props = "GF7035 3-blades" if cy < y_boundary else "GF8040 3-blades"
        else:
            # Fallback: sequential label tracking (original behaviour)
            combined = " ".join(texts)
            if re.search(r"GF\s*7035", combined, re.IGNORECASE):
                current_props = "GF7035 3-blades"
            elif re.search(r"GF\s*8040", combined, re.IGNORECASE):
                current_props = "GF8040 3-blades"
            props = current_props

        row_data = try_parse_data_row(texts)
        if row_data and props:
            row_data["motor_type"] = motor_type or "Unknown"
            row_data["props"] = props
            data.append(row_data)

    if not data:
        sys.exit(
            "No data rows found. Check that the image is a clear photo/scan of the "
            "motor performance table and that OCR can read the text."
        )

    # Reorder columns and sort by prop section then throttle ascending
    cols = ["motor_type", "props", "throttle_pct", "voltage_v", "current_a",
            "thrust_g", "rpm", "power_w", "efficiency_g_per_w"]
    df = pd.DataFrame(data)[cols]
    df = df.sort_values(["props", "throttle_pct"]).reset_index(drop=True)
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
