"""
Extracts drone motor performance test data from:
  - Table images (.jpg, .png, .webp)  → Vision LLM via Ollama (llama3.2-vision)
  - Structured CSV / text files       → pandas parser
  - Web page URLs                     → HTML table scraper (pd.read_html)

Usage:
    python parse_motor_image.py [source] [--output path] [--ollama-model model]

    source  Image path, CSV/text path, or URL.
            Defaults to data/images/motor_table.jpg.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, model_validator, ValidationError

DEFAULT_SOURCE = Path("data/images/motor_table.jpg")
OUTPUT_CSV = Path("data/motor_test_points.csv")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
OUTPUT_COLS = [
    "motor_type", "props", "throttle_pct", "voltage_v", "current_a",
    "thrust_g", "rpm", "power_w", "efficiency_g_per_w",
]

ROW_RANGES = {
    "throttle_pct":       (35,   105),
    "voltage_v":          (10.0,  35.0),
    "current_a":          (0.5,  200.0),
    "thrust_g":           (50,   8000),
    "rpm":                (1000, 50000),
    "power_w":            (5,    8000),
    "efficiency_g_per_w": (0.3,  20.0),
}

_COL_ALIASES = {
    "throttle": "throttle_pct", "throttle%": "throttle_pct", "throttle(%)": "throttle_pct",
    "voltage": "voltage_v", "voltage(v)": "voltage_v",
    "current": "current_a", "current(a)": "current_a",
    "thrust": "thrust_g", "thrust(g)": "thrust_g",
    "power": "power_w", "power(w)": "power_w",
    "efficiency": "efficiency_g_per_w", "eff": "efficiency_g_per_w",
    "g/w": "efficiency_g_per_w", "eff(g/w)": "efficiency_g_per_w",
}


class MotorTestPoint(BaseModel):
    motor_type: str = "Unknown"
    props: str = "Unknown"
    throttle_pct: Optional[int] = None
    voltage_v: float
    current_a: float
    thrust_g: int
    rpm: Optional[int] = None
    power_w: float
    efficiency_g_per_w: Optional[float] = None

    @model_validator(mode="after")
    def check_power(self):
        expected = self.voltage_v * self.current_a
        if abs(self.power_w - expected) / max(expected, 1) > 0.30:
            raise ValueError(
                f"Power {self.power_w:.1f}W ≠ V×A={expected:.1f}W"
            )
        return self


# ---------------------------------------------------------------------------
# Source router
# ---------------------------------------------------------------------------

def parse_source(source: str, model: str = "qwen2.5vl:latest", debug: bool = False) -> pd.DataFrame:
    if source.startswith(("http://", "https://")):
        return parse_url(source)
    path = Path(source)
    if not path.exists():
        sys.exit(f"File not found: {source}")
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return parse_image(path, model, debug=debug)
    return parse_text_file(path)


# ---------------------------------------------------------------------------
# Image parser — Vision LLM via Ollama
# ---------------------------------------------------------------------------

_VISION_PROMPT = """
You are reading a drone motor performance test data table from this image.

Extract EVERY data row and return {"rows": [ ... ]} JSON.
Each element must have EXACTLY these keys — no others:
  "motor_type"         string  motor model label (e.g. "AF310 KV1210")
  "props"              string  propeller label (e.g. "GF7035 3-blades")
  "throttle_pct"       int or null
  "voltage_v"          float   voltage in Volts
  "current_a"          float   current in Amps
  "thrust_g"           int     thrust in grams
  "rpm"                int or null
  "power_w"            float   power in Watts
  "efficiency_g_per_w" float or null

Return ONLY the JSON object, no markdown, no explanation.

Example:
{"rows": [
  {"motor_type":"AF310 KV1210","props":"GF7035 3-blades","throttle_pct":40,"voltage_v":23.71,"current_a":6.38,"thrust_g":752,"rpm":10319,"power_w":151.27,"efficiency_g_per_w":4.97},
  {"motor_type":"AF310 KV1210","props":"GF7035 3-blades","throttle_pct":45,"voltage_v":23.69,"current_a":9.48,"thrust_g":902,"rpm":11170,"power_w":224.58,"efficiency_g_per_w":4.02}
]}
""".strip()


def parse_image(image_path: Path, model: str = "qwen2.5vl:latest", debug: bool = False) -> pd.DataFrame:
    print(f"Parsing image: {image_path}  model={model}")
    try:
        import ollama
    except ImportError:
        sys.exit("ollama package not installed. Run: pip install ollama")
    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": _VISION_PROMPT,
                "images": [str(image_path.resolve())],
            }],
            options={"temperature": 0, "num_predict": -1, "num_ctx": 16384},
        )
        raw = response["message"]["content"]
    except Exception as e:
        sys.exit(
            f"Ollama error: {e}\n"
            f"Ensure ollama is running and '{model}' is available:\n"
            f"  ollama pull {model}"
        )
    return _parse_llm_json(raw, source=str(image_path), debug=debug)


# ---------------------------------------------------------------------------
# Text / CSV parser
# ---------------------------------------------------------------------------

def parse_text_file(path: Path) -> pd.DataFrame:
    print(f"Parsing text file: {path}")
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        df.rename(columns=_COL_ALIASES, inplace=True)
        if {"voltage_v", "current_a", "power_w"}.issubset(df.columns):
            if "motor_type" not in df.columns:
                df["motor_type"] = "Unknown"
            if "props" not in df.columns:
                df["props"] = "Unknown"
            return _validate_df(df)
    except Exception:
        pass
    return _parse_unstructured_lines(path.read_text(encoding="utf-8", errors="replace"))


def _parse_unstructured_lines(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        nums = [float(m) for m in re.findall(r"\d+\.?\d*", line)]
        row = _match_numeric_row(nums)
        if row:
            rows.append(row)
    if not rows:
        sys.exit("No parseable motor data rows found in the text file.")
    return _validate_df(pd.DataFrame(rows))


def _match_numeric_row(nums: list[float]) -> Optional[dict]:
    """Heuristic: find V, A, Thrust, Power by physics (P ≈ V×A)."""
    if len(nums) < 4:
        return None
    vlo, vhi = ROW_RANGES["voltage_v"]
    alo, ahi = ROW_RANGES["current_a"]
    v_idx = next((i for i, n in enumerate(nums) if vlo <= n <= vhi), None)
    if v_idx is None:
        return None
    v = nums[v_idx]
    rest = nums[v_idx + 1:]
    a = next((n for n in rest if alo <= n <= ahi), None)
    if a is None:
        return None
    expected = v * a
    pw = next(
        (n for n in rest
         if ROW_RANGES["power_w"][0] <= n <= ROW_RANGES["power_w"][1]
         and abs(n - expected) / max(expected, 1) < 0.25),
        None,
    )
    if pw is None:
        return None
    thrust = next(
        (n for n in rest
         if ROW_RANGES["thrust_g"][0] <= n <= ROW_RANGES["thrust_g"][1] and n != pw),
        None,
    )
    if thrust is None:
        return None
    throttle = next(
        (nums[i] for i in range(v_idx)
         if ROW_RANGES["throttle_pct"][0] <= nums[i] <= ROW_RANGES["throttle_pct"][1]
         and nums[i] % 5 == 0),
        None,
    )
    rpm = next(
        (n for n in rest
         if ROW_RANGES["rpm"][0] <= n <= ROW_RANGES["rpm"][1] and n not in (pw, thrust)),
        None,
    )
    eff = next(
        (n for n in rest if ROW_RANGES["efficiency_g_per_w"][0] <= n <= ROW_RANGES["efficiency_g_per_w"][1]),
        None,
    )
    return {
        "motor_type": "Unknown", "props": "Unknown",
        "throttle_pct": int(throttle) if throttle is not None else None,
        "voltage_v": round(v, 2), "current_a": round(a, 2),
        "thrust_g": int(thrust),
        "rpm": int(rpm) if rpm is not None else None,
        "power_w": round(pw, 2),
        "efficiency_g_per_w": round(eff, 2) if eff is not None else None,
    }


# ---------------------------------------------------------------------------
# URL parser — HTML tables via pandas
# ---------------------------------------------------------------------------

def parse_url(url: str) -> pd.DataFrame:
    print(f"Fetching HTML tables from: {url}")
    try:
        tables = pd.read_html(url)
    except ImportError:
        sys.exit("lxml required for URL parsing: pip install lxml")
    except Exception as e:
        sys.exit(f"Failed to fetch tables: {e}")
    for i, df in enumerate(tables):
        df.columns = [str(c).strip().lower() for c in df.columns]
        df.rename(columns=_COL_ALIASES, inplace=True)
        if {"voltage_v", "current_a", "power_w"}.issubset(df.columns):
            print(f"  Using table index {i}")
            if "motor_type" not in df.columns:
                df["motor_type"] = "Unknown"
            if "props" not in df.columns:
                df["props"] = "Unknown"
            return _validate_df(df)
    sys.exit(
        "No motor performance table found at URL.\n"
        "The page needs a table with Voltage, Current, and Power columns."
    )


# ---------------------------------------------------------------------------
# Shared: LLM JSON → DataFrame, DataFrame validation
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str, source: str = "", debug: bool = False) -> pd.DataFrame:
    if debug:
        print(f"\n--- RAW LLM OUTPUT ---\n{raw}\n--- END ---\n")
    # Strip markdown fences in case the model wrapped output anyway
    text = re.sub(r"```[^\n]*\n?", "", raw).strip()
    # Extract the first {...} block (handles leading/trailing prose)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        sys.exit(f"No JSON object found in LLM output for {source}.\nOutput:\n{raw[:500]}")
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError as e:
        sys.exit(f"LLM returned invalid JSON: {e}\nOutput:\n{m.group()[:500]}")
    # Unwrap {"rows": [...]} or similar wrapper; fall back to treating dict as single row
    if isinstance(data, dict):
        rows_list = next((v for v in data.values() if isinstance(v, list)), None)
        if rows_list is None:
            rows_list = [data]  # single row returned as flat dict
        data = rows_list
    if not isinstance(data, list):
        sys.exit(f"Expected JSON array from LLM, got: {type(data).__name__}")
    print(f"  LLM returned {len(data)} row(s)")
    rows = []
    errors: list[str] = []
    for i, item in enumerate(data):
        try:
            rows.append(MotorTestPoint(**_non_none(item)).model_dump())
        except (ValidationError, TypeError) as e:
            errors.append(f"  Row {i}: {e}  raw={item}")
    if not rows:
        print("\n--- VALIDATION ERRORS (first 5) ---")
        for msg in errors[:5]:
            print(msg)
        sys.exit(f"\nNo valid rows extracted from {source}.")
    if errors:
        print(f"  Warning: {len(errors)} row(s) dropped (failed validation)")
    return pd.DataFrame(rows)


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    errors: list[str] = []
    for idx, row in df.iterrows():
        d = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        try:
            rows.append(MotorTestPoint(**_non_none(d)).model_dump())
        except (ValidationError, TypeError) as e:
            errors.append(f"  Row {idx}: {e}\n    data={d}")
    if not rows:
        print("\n--- VALIDATION ERRORS (first 5) ---")
        for msg in errors[:5]:
            print(msg)
        sys.exit("\nAll rows failed validation (range checks or P≠V×A).")
    if errors:
        print(f"  Warning: {len(errors)} row(s) dropped (failed validation)")
    return pd.DataFrame(rows)


def _non_none(d: dict) -> dict:
    """Return only keys present in MotorTestPoint where value is not None."""
    fields = set(MotorTestPoint.model_fields)
    return {k: v for k, v in d.items() if k in fields and v is not None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract motor performance data from an image, CSV, or URL"
    )
    parser.add_argument(
        "source", nargs="?", default=str(DEFAULT_SOURCE),
        help=f"Image path, CSV/text path, or URL (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--ollama-model", default="qwen2.5vl:latest",
        help="Ollama vision model for image parsing (default: qwen2.5vl:latest)",
    )
    parser.add_argument("--debug", action="store_true", help="Print raw LLM output")
    args = parser.parse_args()

    df = parse_source(args.source, model=args.ollama_model, debug=args.debug)
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLS]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows → {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
