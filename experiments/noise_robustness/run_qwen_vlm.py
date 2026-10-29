
import argparse
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

ROOT = Path("experiments/noise_robustness")
METADATA_PATH = ROOT / "noise_metadata.json"
OUTPUT_PATH = ROOT / "results/qwen_vlm_mixed_135.json"

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
FIELDS = [
    "load_number", "pickup_location", "pickup_time",
    "dropoff_location", "dropoff_time",
    "total_rate", "rate_per_mile"
]

# Exact prompt from the original VLM recovery script.
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


def image_to_data_url(path):
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def parse_output(raw):
    text = raw.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    # Original recovery script's load-number JSON repair.
    text = re.sub(
        r'("load_number"\s*:\s*)([A-Za-z0-9\-]+)(\s*[,}])',
        r'\1"\2"\3',
        text
    )

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def save_results(rows):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    tmp.replace(OUTPUT_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=135)
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found")

    client = InferenceClient(
        api_key=token,
        provider="featherless-ai"
    )

    metadata = json.loads(METADATA_PATH.read_text())
    if len(metadata) != 135:
        raise ValueError("Expected 135 noisy images")

    if OUTPUT_PATH.exists():
        results = json.loads(OUTPUT_PATH.read_text())
    else:
        results = []

    completed = {
        (r["split"], r["source_image"])
        for r in results
        if r.get("_status") == "ok"
    }

    for index, row in enumerate(metadata[:args.limit], 1):
        split = row["split"]
        image_name = row["source_image"]
        key = (split, image_name)

        if key in completed:
            continue

        image_path = Path(row["noisy_path"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        prompt = PROMPT_TEMPLATE.format(image_name=image_name)
        image_url = image_to_data_url(image_path)

        result = {
            "split": split,
            "source_image": image_name,
            **{field: None for field in FIELDS}
        }

        error = None

        for attempt in range(1, 6):
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_url}
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )

                raw = completion.choices[0].message.content
                parsed = parse_output(raw)

                for field in FIELDS:
                    result[field] = parsed.get(field)

                result["_status"] = "ok"
                error = None
                break

            except Exception as exc:
                error = str(exc)
                print(
                    f"[{index}/{args.limit}] "
                    f"{split}/{image_name} "
                    f"attempt {attempt} failed: {error}",
                    flush=True
                )

                if "model_not_supported" in error:
                    break

                if attempt < 5:
                    time.sleep(3 * attempt)

        if error:
            result["_status"] = "failed"
            result["_error"] = error

        results = [
            r for r in results
            if (r["split"], r["source_image"]) != key
        ]
        results.append(result)
        save_results(results)

        print(
            f"[{index}/{args.limit}] "
            f"{split}/{image_name}: {result['_status']}",
            flush=True
        )

        time.sleep(2)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
