import json
import re

RAW_PATH = "results/qwen_vlm_160.json"
OUT_PATH = "results/qwen_vlm_160_clean.json"
FAILED_PATH = "results/qwen_vlm_160_failed.json"

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

def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def fix_common_json_issues(text: str) -> str:
    # quote unquoted load_number values like: "load_number": 111C2HNDW
    text = re.sub(
        r'("load_number"\s*:\s*)([A-Za-z0-9\-]+)(\s*[,}])',
        r'\1"\2"\3',
        text
    )
    return text

def empty_record(split, image_name):
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

with open(RAW_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []
failed = []

for item in data:
    split = item["split"]
    image_name = item["source_image"]

    if "error" in item:
        failed.append(item)
        cleaned.append(empty_record(split, image_name))
        continue

    raw = item.get("raw_output", "").strip()
    raw = strip_code_fences(raw)
    raw = fix_common_json_issues(raw)

    try:
        parsed = json.loads(raw)
        rec = empty_record(split, image_name)
        for field in FIELDS:
            if field in parsed:
                rec[field] = parsed[field]
        # force true filename from outer record
        rec["source_image"] = image_name
        rec["split"] = split
        cleaned.append(rec)
    except Exception as e:
        failed.append({
            "split": split,
            "source_image": image_name,
            "error": f"parse_error: {e}",
            "raw_output": item.get("raw_output", "")
        })
        cleaned.append(empty_record(split, image_name))

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, ensure_ascii=False, indent=2)

with open(FAILED_PATH, "w", encoding="utf-8") as f:
    json.dump(failed, f, ensure_ascii=False, indent=2)

print(f"Saved cleaned file: {OUT_PATH}")
print(f"Saved failed list:   {FAILED_PATH}")
print(f"Total: {len(data)} | Failed/parsing issues: {len(failed)}")