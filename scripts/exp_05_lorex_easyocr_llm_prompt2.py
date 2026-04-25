# scripts/exp_05_easyocr_llm_all160_prompt2.py

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import easyocr

# =========================
# CONFIG
# =========================
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

DATASET_ROOT = Path("../LoREx-160/rate_confirmations")
IMAGES_ROOT = DATASET_ROOT / "images"

OUT_PATH = Path("results/easyocr_llm_160_prompt2.json")
FAILED_PATH = Path("results/easyocr_llm_160_prompt2_failed.json")

SPLITS = ["train", "val", "test"]
MAX_RETRIES = 3

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto",
)

reader = easyocr.Reader(["en"], gpu=False)

FIELDS = [
    "split",
    "source_image",
    "load_number",
    "pickup_location",
    "pickup_time",
    "dropoff_location",
    "dropoff_time",
    "total_rate",
    "rate_per_mile",
]

# =========================
# OCR
# =========================
def ocr_easy(img_path: Path) -> str:
    result = reader.readtext(str(img_path), detail=0)
    lines = [str(x).strip() for x in result if str(x).strip()]
    return "\n".join(lines)

# =========================
# PROMPT
# =========================
def build_prompt(image_name: str, ocr_text: str) -> str:
    return f"""
You are extracting structured fields from OCR text of a logistics rate confirmation or dispatch screenshot.

Return ONLY valid JSON with exactly these keys:
{{
  "source_image": "{image_name}",
  "load_number": null,
  "pickup_location": null,
  "pickup_time": null,
  "dropoff_location": null,
  "dropoff_time": null,
  "total_rate": null,
  "rate_per_mile": null
}}

Important guidance:
- The load_number may NOT be explicitly labeled as "load number".
- It may appear as a short standalone alphanumeric code near the top of the document or screenshot.
- It may look like examples such as: 116WQ2Q4S, T-112K9S5H5, 111C2HNDW.
- If you see a plausible shipment/load identifier in the OCR text, extract it exactly as written.
- Do not set load_number to null if a plausible alphanumeric identifier is present.
- Keep operational codes like IND2, MDW5, RDG1, ELP1 inside location fields when they belong there, but do not confuse them with the load number.
- pickup_location and dropoff_location should be the location strings.
- pickup_time and dropoff_time should be the time/date strings.
- total_rate should be the full money amount for the load.
- rate_per_mile should be the $.../mi value.

Rules:
- Use only information present in the OCR text.
- Do not invent values.
- Preserve formatting as closely as possible.
- Return JSON only, with no markdown and no explanation.
- Use null if truly missing.

OCR text:
\"\"\"
{ocr_text}
\"\"\"
""".strip()

# =========================
# LLM
# =========================
def call_llm(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0,
    )
    return completion.choices[0].message.content

# =========================
# HELPERS
# =========================
def clean_json_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def extract_json(text: str):
    text = clean_json_text(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("No JSON found")

def empty_record(split: str, image_name: str):
    return {
        "split": split,
        "source_image": image_name,
        "load_number": None,
        "pickup_location": None,
        "pickup_time": None,
        "dropoff_location": None,
        "dropoff_time": None,
        "total_rate": None,
        "rate_per_mile": None,
    }

def run_with_retry(prompt, image_name):
    for i in range(MAX_RETRIES):
        try:
            print(f"[{image_name}] attempt {i+1}")
            return call_llm(prompt)
        except Exception as e:
            print(f"[{image_name}] retry error: {e}")
            time.sleep(2)
    return None

def safe_key(p: Path):
    try:
        return int(p.stem)
    except Exception:
        return p.stem.lower()

def list_image_files(split_dir: Path):
    return sorted(
        [
            p for p in split_dir.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ],
        key=safe_key
    )

# =========================
# MAIN
# =========================
def main():
    results = []
    failed = []

    for split in SPLITS:
        split_dir = IMAGES_ROOT / split

        if not split_dir.exists():
            print(f"[WARN] Missing split dir: {split_dir}")
            continue

        image_files = list_image_files(split_dir)
        print(f"\nProcessing {split}: {len(image_files)} images")

        for img_path in image_files:
            image_name = img_path.name

            try:
                ocr_text = ocr_easy(img_path)
            except Exception as e:
                results.append(empty_record(split, image_name))
                failed.append({
                    "split": split,
                    "img": image_name,
                    "stage": "ocr",
                    "err": str(e)
                })
                continue

            prompt = build_prompt(image_name, ocr_text)
            raw = run_with_retry(prompt, f"{split}/{image_name}")

            if not raw:
                results.append(empty_record(split, image_name))
                failed.append({
                    "split": split,
                    "img": image_name,
                    "stage": "llm",
                    "err": "no response",
                    "ocr_text": ocr_text
                })
                continue

            try:
                parsed = extract_json(raw)
                rec = empty_record(split, image_name)
                rec.update(parsed)
                rec["split"] = split
                rec["source_image"] = image_name
                results.append(rec)
                print(f"[OK] {split}/{image_name} -> {rec.get('load_number')}")
            except Exception as e:
                results.append(empty_record(split, image_name))
                failed.append({
                    "split": split,
                    "img": image_name,
                    "stage": "parse",
                    "err": str(e),
                    "ocr_text": ocr_text,
                    "raw_output": raw
                })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(FAILED_PATH, "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {OUT_PATH}")
    print(f"Saved failed: {FAILED_PATH}")
    print(f"Total: {len(results)} | Failed: {len(failed)}")

if __name__ == "__main__":
    main()