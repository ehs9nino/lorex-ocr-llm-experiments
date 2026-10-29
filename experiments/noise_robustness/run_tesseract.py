
import json
import time
from pathlib import Path

import pytesseract
from PIL import Image

ROOT = Path("experiments/noise_robustness")
OUTPUT = ROOT / "results/ocr/tesseract_mixed_135.json"

records = json.loads(
    (ROOT / "noise_metadata.json").read_text(encoding="utf-8")
)

results = []

for i, record in enumerate(records, 1):
    start = time.perf_counter()

    try:
        text = pytesseract.image_to_string(
            Image.open(record["noisy_path"]),
            config="--oem 3 --psm 6"
        )

        item = {
            "split": record["split"],
            "source_image": record["source_image"],
            "noise_type": record["noise_type"],
            "ocr_text": text,
            "time_seconds": round(
                time.perf_counter() - start, 3
            ),
        }

        status = "OK"

    except Exception as e:
        item = {
            "split": record["split"],
            "source_image": record["source_image"],
            "noise_type": record["noise_type"],
            "error": str(e),
        }

        status = "FAILED"

    results.append(item)

    OUTPUT.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"[{i}/135] {record['source_image']} {status}", flush=True)

print(f"\nCompleted. Saved: {OUTPUT}")
