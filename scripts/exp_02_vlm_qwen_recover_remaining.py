import os
import json
import time
import base64
import mimetypes
import re
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

# Path to local dataset images
DATASET_ROOT = Path("../LoREx-160/rate_confirmations/images")

# Existing raw results file
RAW_RESULTS_PATH = Path("results/qwen_vlm_160.json")

# Output files
OUT_COMPLETED_PATH = Path("results/qwen_vlm_160_completed.json")
OUT_STILL_FAILED_PATH = Path("results/qwen_vlm_160_still_failed.json")

# Main model
PRIMARY_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Optional fallback model if Qwen still fails.
# Keep as None if you want pure Qwen only.
FALLBACK_MODEL = None

MAX_RETRIES = 5
SLEEP_BETWEEN_ATTEMPTS = 3
SLEEP_BETWEEN_IMAGES = 2

FIELDS = [
    "source_image",
    "load_number",
    "pickup_location",
    "pickup_time",
    "dropoff_location",
    "dropoff_time",
    "total_rate",
    "rate_per_mile",
]

PROMPT_TEMPLATE = """
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


# =========================
# HELPERS
# =========================
def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def fix_common_json_issues(text: str) -> str:
    # quote unquoted load_number values like:
    # "load_number": 111C2HNDW
    text = re.sub(
        r'("load_number"\s*:\s*)([A-Za-z0-9\-]+)(\s*[,}])',
        r'\1"\2"\3',
        text
    )
    return text


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


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def parse_output(raw_output: str, split: str, image_name: str):
    raw_output = strip_code_fences(raw_output)
    raw_output = fix_common_json_issues(raw_output)

    parsed = json.loads(raw_output)

    rec = empty_record(split, image_name)
    for field in FIELDS:
        if field in parsed:
            rec[field] = parsed[field]

    # enforce outer metadata
    rec["split"] = split
    rec["source_image"] = image_name
    return rec


def call_model(model_name: str, image_path: Path, image_name: str) -> str:
    prompt = PROMPT_TEMPLATE.format(image_name=image_name)
    image_data_url = image_to_data_url(image_path)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        max_tokens=300,
    )

    return completion.choices[0].message.content


def try_model_with_retry(model_name: str, split: str, image_name: str, image_path: Path):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[{model_name}] {split}/{image_name} attempt {attempt}")
            raw_output = call_model(model_name, image_path, image_name)
            parsed = parse_output(raw_output, split, image_name)
            parsed["_recovered_with"] = model_name
            return parsed
        except Exception as e:
            last_error = str(e)
            print(f"  failed: {last_error}")

            # hard provider problem: don't waste retries
            if "model_not_supported" in last_error:
                break

            time.sleep(SLEEP_BETWEEN_ATTEMPTS * attempt)

    return {
        "split": split,
        "source_image": image_name,
        "error": last_error or "failed_after_retries"
    }


def recover_one_failed_record(row):
    split = row["split"]
    image_name = row["source_image"]
    image_path = DATASET_ROOT / split / image_name

    if not image_path.exists():
        return {
            "split": split,
            "source_image": image_name,
            "error": f"local_file_missing: {image_path}"
        }

    # First try Qwen again
    result = try_model_with_retry(PRIMARY_MODEL, split, image_name, image_path)
    if "error" not in result:
        return result

    # Optional fallback
    if FALLBACK_MODEL:
        print(f"Trying fallback for {split}/{image_name}")
        result_fb = try_model_with_retry(FALLBACK_MODEL, split, image_name, image_path)
        if "error" not in result_fb:
            return result_fb

    return result


# =========================
# MAIN
# =========================
def main():
    if not RAW_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing input results file: {RAW_RESULTS_PATH}")

    with open(RAW_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Deduplicate by (split, source_image), keeping the last occurrence
    dedup = {}
    for row in data:
        key = (row["split"], row["source_image"])
        dedup[key] = row

    full_rows = list(dedup.values())

    total = len(full_rows)
    failed_rows = [row for row in full_rows if "error" in row]

    print(f"Unique records found: {total}")
    print(f"Failed records to retry: {len(failed_rows)}")

    recovered_count = 0
    still_failed = []

    for row in failed_rows:
        split = row["split"]
        image_name = row["source_image"]

        print(f"\n=== Recovering {split}/{image_name} ===")
        new_row = recover_one_failed_record(row)

        key = (split, image_name)
        dedup[key] = new_row

        if "error" in new_row:
            still_failed.append(new_row)
        else:
            recovered_count += 1

        # save progress after each image
        current_rows = sorted(
            dedup.values(),
            key=lambda r: (r["split"], int(Path(r["source_image"]).stem))
        )

        with open(OUT_COMPLETED_PATH, "w", encoding="utf-8") as f:
            json.dump(current_rows, f, ensure_ascii=False, indent=2)

        with open(OUT_STILL_FAILED_PATH, "w", encoding="utf-8") as f:
            json.dump(still_failed, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_BETWEEN_IMAGES)

    final_rows = sorted(
        dedup.values(),
        key=lambda r: (r["split"], int(Path(r["source_image"]).stem))
    )

    final_success = sum(1 for r in final_rows if "error" not in r)
    final_failed = sum(1 for r in final_rows if "error" in r)

    with open(OUT_COMPLETED_PATH, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=2)

    with open(OUT_STILL_FAILED_PATH, "w", encoding="utf-8") as f:
        json.dump([r for r in final_rows if "error" in r], f, ensure_ascii=False, indent=2)

    print("\n=========================")
    print(f"Saved completed file: {OUT_COMPLETED_PATH}")
    print(f"Saved still-failed file: {OUT_STILL_FAILED_PATH}")
    print(f"Recovered this run: {recovered_count}")
    print(f"Final success: {final_success}/{len(final_rows)}")
    print(f"Final failed: {final_failed}/{len(final_rows)}")


if __name__ == "__main__":
    main()