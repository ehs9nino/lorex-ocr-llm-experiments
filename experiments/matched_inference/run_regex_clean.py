
import json
import sys
from pathlib import Path

ROOT = Path("experiments/noise_robustness")
INPUT = Path("experiments/matched_inference/ocr/clean/tesseract_clean_135.json")
OUTPUT = Path("experiments/matched_inference/predictions/clean/regex_tesseract_clean_135.json")

# Import the original regex extractor without modifying it.
sys.path.insert(0, str(Path("scripts").resolve()))

from exp_01_lorex_regex_tesseract import (
    extract_regex_fields,
    FIELDS,
)

records = json.loads(INPUT.read_text(encoding="utf-8"))

# Ensure all 135 image IDs are unique.
keys = [(r["split"], r["source_image"]) for r in records]
assert len(records) == 135, f"Expected 135, got {len(records)}"
assert len(set(keys)) == 135, "Duplicate image IDs found"

results = []

for index, record in enumerate(records, 1):
    text = record.get("ocr_text", "")

    # Identical extraction function to the original benchmark.
    prediction = extract_regex_fields(text)

    result = {
        "split": record["split"],
        "source_image": record["source_image"],
    }

    for field in FIELDS:
        result[field] = prediction.get(field, "")

    results.append(result)

    if index % 25 == 0 or index == len(records):
        print(f"Processed {index}/{len(records)}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nCompleted: {len(results)} predictions")
print(f"Saved: {OUTPUT}")
