
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import json
import time
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("experiments/noise_robustness")
IMAGES = ROOT / "images/mixed"
METADATA = ROOT / "noise_metadata.json"
OUTPUT = ROOT / "results/ocr"

OUTPUT.mkdir(parents=True, exist_ok=True)

with open(METADATA, encoding="utf-8") as f:
    records = json.load(f)

def extract_paddle(result):
    lines = []
    for page in result or []:
        lines.extend(str(x) for x in page.get("rec_texts", []))
    return "\n".join(lines)

# Initialize installed OCR engines
engines = {}

try:
    import pytesseract
    engines["tesseract"] = lambda p: pytesseract.image_to_string(
        Image.open(p), config="--oem 3 --psm 6"
    )
except Exception as e:
    print("Tesseract unavailable:", e)

try:
    from paddleocr import PaddleOCR
    paddle = PaddleOCR(use_textline_orientation=True, lang="en")
    engines["paddle"] = lambda p: extract_paddle(
        paddle.predict(str(p))
    )
except Exception as e:
    print("PaddleOCR unavailable:", e)

try:
    import easyocr
    easy = easyocr.Reader(["en"], gpu=False)
    engines["easyocr"] = lambda p: "\n".join(
        easy.readtext(str(p), detail=0)
    )
except Exception as e:
    print("EasyOCR unavailable:", e)

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    doctr_model = ocr_predictor(pretrained=True)
    def run_doctr(p):
        result = doctr_model(DocumentFile.from_images(str(p)))
        data = result.export()
        lines = []
        for page in data["pages"]:
            for block in page["blocks"]:
                for line in block["lines"]:
                    lines.append(" ".join(
                        word["value"] for word in line["words"]
                    ))
        return "\n".join(lines)
    engines["doctr"] = run_doctr
except Exception as e:
    print("DocTR unavailable:", e)

# OpenOCR has version-dependent APIs.
# We will add its adapter after verifying your existing implementation.

print("Available engines:", list(engines))

for engine_name, run_ocr in engines.items():
    out_file = OUTPUT / f"{engine_name}_mixed_135.json"

    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    completed = {
        (r["split"], r["source_image"])
        for r in results if "ocr_text" in r
    }

    for index, record in enumerate(records, 1):
        key = (record["split"], record["source_image"])
        if key in completed:
            continue

        image_path = Path(record["noisy_path"])
        start = time.perf_counter()

        try:
            text = run_ocr(image_path)
            item = {
                "split": record["split"],
                "source_image": record["source_image"],
                "noise_type": record["noise_type"],
                "ocr_text": text,
                "time_seconds": round(
                    time.perf_counter() - start, 3
                ),
            }
        except Exception as e:
            item = {
                "split": record["split"],
                "source_image": record["source_image"],
                "noise_type": record["noise_type"],
                "error": str(e),
            }

        results.append(item)

        # Save after each image to support resuming
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(
            f"[{engine_name}] {index}/{len(records)} "
            f"{record['source_image']} "
            f"{'OK' if 'ocr_text' in item else 'FAILED'}",
            flush=True
        )

    print("Saved:", out_file)

print("Local OCR batch finished.")
