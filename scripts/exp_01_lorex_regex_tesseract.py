import json
import re
from pathlib import Path

import pytesseract
from PIL import Image

# =========================
# CONFIG
# =========================
DATASET_ROOT = Path("../LoREx-160/rate_confirmations")
IMAGES_ROOT = DATASET_ROOT / "images"

OUT_JSON = Path("results/regex_tesseract_160.json")

FIELDS = [
    "load_number",
    "pickup_location",
    "pickup_time",
    "dropoff_location",
    "dropoff_time",
    "total_rate",
    "rate_per_mile",
]

SPLITS = ["train", "val", "test"]

VALID_EXTS = {".jpg", ".jpeg", ".png"}

# =========================
# OCR
# =========================
def ocr_tesseract(img_path: Path) -> str:
    img = Image.open(img_path).convert("RGB")
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6")

# =========================
# REGEX EXTRACTION
# =========================
def extract_regex_fields(text: str) -> dict:
    out = {f: "" for f in FIELDS}

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    t = " ".join(lines)
    t = re.sub(r"\s+", " ", t).strip()

    load_patterns = [
        r"\bT-[A-Z0-9]{8,12}\b",
        r"\b[A-Z0-9]{8,12}\b",
    ]
    for pat in load_patterns:
        m = re.search(pat, t)
        if m:
            cand = m.group(0)
            if cand not in {"EDT", "CDT", "MDT", "PDT", "MST"}:
                out["load_number"] = cand
                break

    money_matches = re.findall(r"\$\s?\d[\d\s,\.]*(?:/\s*mi)?", t, flags=re.I)
    money_matches = [m.strip() for m in money_matches if m.strip()]

    rpm_match = re.search(r"\$\s?\d[\d\s,\.]*/\s*mi", t, flags=re.I)
    if rpm_match:
        out["rate_per_mile"] = re.sub(r"\s+", "", rpm_match.group(0))

    for m in money_matches:
        if "/mi" not in m.lower():
            out["total_rate"] = m
            break

    loc_patterns = [
        r"\b[A-Z0-9]{3,5}\s+[A-Z][A-Z\s]+,\s*[A-Z]{2}\s*\d{5}\b",
        r"\b[A-Z0-9]{3,5}\s+[A-Z][A-Za-z\s]+,\s*[A-Za-z]+\s*\d{5}\b",
        r"\b[A-Z0-9]{3,5}\s+[A-Z][A-Za-z\s]+,\s*[A-Za-z]+\b",
    ]

    found_locs = []
    for pat in loc_patterns:
        for m in re.finditer(pat, t):
            val = re.sub(r"\s+", " ", m.group(0)).strip()
            if val not in found_locs:
                found_locs.append(val)

    if len(found_locs) >= 1:
        out["pickup_location"] = found_locs[0]
    if len(found_locs) >= 2:
        out["dropoff_location"] = found_locs[1]

    time_pat = r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+[A-Z][a-z]{2}\s+\d{1,2},\s+\d{1,2}:\d{2}\s+(?:EDT|CDT|MDT|PDT|MST)\b"
    found_times = re.findall(time_pat, t)

    if len(found_times) >= 1:
        out["pickup_time"] = found_times[0]
    if len(found_times) >= 2:
        out["dropoff_time"] = found_times[1]

    return out

# =========================
# HELPERS
# =========================
def safe_key(p: Path):
    try:
        return int(p.stem)
    except Exception:
        return p.stem.lower()

def list_image_files(split_dir: Path):
    return sorted(
        [p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS],
        key=safe_key
    )

# =========================
# MAIN
# =========================
def main():
    if not IMAGES_ROOT.exists():
        raise FileNotFoundError(f"Images root not found: {IMAGES_ROOT}")

    results_json = []
    total_expected = 0

    for split in SPLITS:
        split_dir = IMAGES_ROOT / split

        if not split_dir.exists():
            print(f"[WARN] Missing split folder: {split_dir}")
            continue

        image_files = list_image_files(split_dir)
        print(f"\nProcessing split: {split} ({len(image_files)} images)")
        total_expected += len(image_files)

        for img_path in image_files:
            image_name = img_path.name

            try:
                text = ocr_tesseract(img_path)
                pred = extract_regex_fields(text)
            except Exception as e:
                print(f"[ERROR] {split}/{image_name}: {e}")
                pred = {k: "" for k in FIELDS}

            result_item = {
                "split": split,
                "source_image": image_name,
            }

            for f in FIELDS:
                result_item[f] = pred.get(f, "")

            results_json.append(result_item)

    print(f"\nExpected images: {total_expected}")
    print(f"Generated records: {len(results_json)}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    print(f"\nSaved extraction JSON: {OUT_JSON}")

if __name__ == "__main__":
    main()