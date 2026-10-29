
import json
import time
from pathlib import Path
from openocr import OpenOCR

ROOT = Path("experiments/noise_robustness")
METADATA = ROOT / "noise_metadata.json"
OUTPUT = ROOT / "results/ocr/openocr_mixed_135.json"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

records = json.loads(
    METADATA.read_text(encoding="utf-8")
)

# Same initialization as original benchmark
ocr = OpenOCR()


# Exact OCR parsing logic from original experiment
def ocr_openocr(img_path: Path) -> str:
    result = ocr.model(
        img_path=str(img_path),
        is_visualize=False
    )

    lines = []

    if not isinstance(result, tuple) or len(result) == 0:
        return ""

    raw_items = result[0]
    if not raw_items:
        return ""

    for raw in raw_items:
        if not isinstance(raw, str):
            continue

        if "\t" not in raw:
            continue

        _, payload = raw.split("\t", 1)

        try:
            parsed = json.loads(payload)
        except Exception:
            continue

        for item in parsed:
            text = str(
                item.get("transcription", "")
            ).strip()

            if text:
                lines.append(text)

    return "\n".join(lines)


# Resume support
if OUTPUT.exists():
    results = json.loads(
        OUTPUT.read_text(encoding="utf-8")
    )
else:
    results = []

completed = {
    (r["split"], r["source_image"])
    for r in results
    if "ocr_text" in r
}

for index, record in enumerate(records, 1):
    key = (record["split"], record["source_image"])

    if key in completed:
        continue

    img_path = Path(record["noisy_path"])
    start = time.perf_counter()

    try:
        text = ocr_openocr(img_path)

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

    results = [
        r for r in results
        if (r["split"], r["source_image"]) != key
    ]
    results.append(item)

    OUTPUT.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        f"[OpenOCR] {index}/{len(records)} "
        f"{record['source_image']} {status}",
        flush=True
    )

print(f"\nCompleted. Saved: {OUTPUT}")
