# scripts/exp_03_tesseract_llm_all160_prompt2.py

import json
import os
import re
import time
from pathlib import Path

import pytesseract
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# =========================
# CONFIG
# =========================
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

DATASET_ROOT = Path("../LoREx-160/rate_confirmations")
IMAGES_ROOT = DATASET_ROOT / "images"

OUT_PATH = Path("results/tesseract_llm_160_prompt2.json")
FAILED_PATH = Path("results/tesseract_llm_160_prompt2_failed.json")

SPLITS = ["train", "val", "test"]
MAX_RETRIES = 3

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto",
)

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
def ocr_tesseract(img_path: Path) -> str:
    img = Image.open(img_path).convert("RGB")
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6")

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

def extract_first_json_object(text: str) -> str:
    text = clean_json_text(text)

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        json.loads(candidate)
        return candidate

    raise ValueError("No valid JSON object found")

def empty_record(split: str, image_name: str) -> dict:
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

def run_with_retry(prompt: str, image_name: str, max_retries: int = MAX_RETRIES) -> str | None:
    for attempt in range(max_retries):
        try:
            print(f"[{image_name}] LLM attempt {attempt + 1}")
            return call_llm(prompt)
        except Exception as e:
            print(f"[{image_name}] failed: {e}")
            time.sleep(3 * (attempt + 1))
    return None

def safe_key(p: Path):
    try:
        return int(p.stem)
    except Exception:
        return p.stem

def list_image_files(split_dir: Path):
    return sorted(
        [p for p in split_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]],
        key=safe_key
    )

# =========================
# MAIN
# =========================
def main():
    if not IMAGES_ROOT.exists():
        raise FileNotFoundError(f"Images root not found: {IMAGES_ROOT}")

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
                ocr_text = ocr_tesseract(img_path)
            except Exception as e:
                print(f"[{split}/{image_name}] OCR failed: {e}")
                results.append(empty_record(split, image_name))
                failed.append({
                    "split": split,
                    "source_image": image_name,
                    "stage": "ocr",
                    "error": str(e),
                })
                continue

            prompt = build_prompt(image_name, ocr_text)
            raw_output = run_with_retry(prompt, f"{split}/{image_name}")

            if raw_output is None:
                results.append(empty_record(split, image_name))
                failed.append({
                    "split": split,
                    "source_image": image_name,
                    "stage": "llm",
                    "error": "max retries exceeded",
                    "ocr_text": ocr_text,
                })
                continue

            try:
                parsed = json.loads(extract_first_json_object(raw_output))
                record = empty_record(split, image_name)
                record.update(parsed)
                record["split"] = split
                record["source_image"] = image_name
                results.append(record)
                print(f"[OK] {split}/{image_name} -> load_number={record.get('load_number')}")
            except Exception as e:
                print(f"[{split}/{image_name}] JSON parse failed: {e}")
                results.append(empty_record(split, image_name))
                failed.append({
                    "split": split,
                    "source_image": image_name,
                    "stage": "parse",
                    "error": str(e),
                    "ocr_text": ocr_text,
                    "raw_output": raw_output,
                })

            time.sleep(2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(FAILED_PATH, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    print(f"\nSaved JSON: {OUT_PATH}")
    print(f"Saved failed JSON: {FAILED_PATH}")
    print(f"Total: {len(results)} | Failed: {len(failed)}")

if __name__ == "__main__":
    main()