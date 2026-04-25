import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# =========================
# CONFIG
# =========================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto",
)

DATASET_ROOT = Path("../LoREx-160/rate_confirmations/images")
OUT_PATH = "results/qwen_vlm_160.json"
MAX_RETRIES = 3

# =========================
# BUILD REAL RECORDS
# =========================
def build_records():
    records = []

    for split in ["val", "train", "test"]:
        split_dir = DATASET_ROOT / split
        if not split_dir.exists():
            print(f"[WARN] Missing folder: {split_dir}")
            continue

        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            files.extend(split_dir.glob(ext))

        files = sorted(files, key=lambda p: int(p.stem))

        for p in files:
            records.append({
                "split": split,
                "source_image": p.name,   # keep real extension
            })

    return records

def build_image_url(split: str, image_name: str) -> str:
    return f"https://raw.githubusercontent.com/ehs9nino/LoREx-160/main/rate_confirmations/images/{split}/{image_name}"

# =========================
# MODEL CALL
# =========================
def run_one(split: str, image_name: str) -> str:
    image_url = build_image_url(split, image_name)

    prompt = f"""
Extract fields from this logistics document.

Return ONLY valid JSON with exactly these keys and no extra text:
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

Rules:
- Copy values exactly as written in the image.
- Do not normalize, reformat, translate, or expand anything.
- Keep location codes such as RDG1, ELP1, MDW5, etc.
- Keep money formatting exactly as shown.
- Keep rate_per_mile exactly as shown, including /mi.
- Keep dates/times exactly as shown.
- If unsure, output null.
- Return JSON only.
""".strip()

    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=300,
    )

    return completion.choices[0].message.content

def run_with_retry(split: str, image_name: str, max_retries: int = MAX_RETRIES):
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"[{split}/{image_name}] attempt {attempt + 1}")
            return run_one(split, image_name)
        except Exception as e:
            last_error = str(e)
            print(f"[{split}/{image_name}] failed: {e}")
            time.sleep(4 * (attempt + 1))

    return {"error": last_error or "failed_after_retries"}

# =========================
# MAIN
# =========================
def main():
    os.makedirs("results", exist_ok=True)

    records = build_records()
    print(f"Found {len(records)} images")

    # resume support
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    # skip only successful ones
    done = {
        (r["split"], r["source_image"])
        for r in results
        if "raw_output" in r
    }

    for rec in records:
        split = rec["split"]
        image_name = rec["source_image"]
        key = (split, image_name)

        if key in done:
            print(f"[SKIP] {split}/{image_name}")
            continue

        output = run_with_retry(split, image_name)

        if isinstance(output, dict) and "error" in output:
            results.append({
                "split": split,
                "source_image": image_name,
                "error": output["error"]
            })
        else:
            results.append({
                "split": split,
                "source_image": image_name,
                "raw_output": output
            })

        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(2)

    print(f"Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()