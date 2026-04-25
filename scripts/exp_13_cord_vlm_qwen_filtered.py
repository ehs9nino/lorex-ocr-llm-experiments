import os
import json
import time
import base64
import mimetypes
import argparse
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

# Start with 7B
MODEL_NAME = "Qwen/Qwen2.5-VL-72B-Instruct:fastest"

client = InferenceClient(api_key=HF_TOKEN)

PROMPT_TEMPLATE = """
Extract structured fields from this receipt image.

Return ONLY valid JSON with exactly these keys:
{
  "item_name": null,
  "item_price": null,
  "sub_total": null,
  "tax": null,
  "total": null
}

Rules:
- Copy values exactly as written in the image.
- Do not normalize commas, dots, spaces, or currency symbols.
- item_name = the first main purchased item, not store name, not payment method, not summary line.
- item_price = the price associated with item_name, not subtotal, tax, cash, or change.
- sub_total = value near keywords like SUBTOTAL, SUB TOTAL, or Sub Total.
- tax = value near keywords like TAX, PB1, or service tax.
- total = final payable amount near keywords like TOTAL, GRAND TOTAL, or Total.
- Do NOT use CASH, CHANGE, CARD PAYMENT, CREDIT CARD, or DISCOUNT as total.
- If multiple totals appear, choose the final payable amount.
- If unsure, output null.
- Return JSON only.
""".strip()


# =========================
# HELPERS
# =========================
def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def parse_llm_json(raw_text: str) -> dict:
    raw_text = strip_code_fences(raw_text)
    return json.loads(raw_text)


def empty_record(image_name: str) -> dict:
    return {
        "source_image": image_name,
        "item_name": None,
        "item_price": None,
        "sub_total": None,
        "tax": None,
        "total": None,
    }


def load_existing(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# =========================
# MODEL CALL
# =========================
def run_one(image_path: Path) -> str:
    image_data_url = image_to_data_url(image_path)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        max_tokens=500,
    )

    return completion.choices[0].message.content


def run_with_retry(image_path: Path, max_retries: int = 4):
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"[{image_path.name}] attempt {attempt + 1}")
            return run_one(image_path)
        except Exception as e:
            last_error = str(e)
            print(f"[{image_path.name}] failed: {e}")
            time.sleep(3 * (attempt + 1))

    return {"error": last_error or "failed_after_retries"}


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Path to local image folder")
    parser.add_argument("--source-list", required=True, help="JSON file used only to get filtered source_image names")
    parser.add_argument("--output", required=True, help="Path to save VLM results")
    args = parser.parse_args()

    img_dir = Path(args.image_dir)
    source_list_path = Path(args.source_list)
    out_path = Path(args.output)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")

    with open(source_list_path, "r", encoding="utf-8") as f:
        source_list = json.load(f)

    image_names = [rec["source_image"] for rec in source_list]
    image_paths = [img_dir / name for name in image_names]

    results = load_existing(out_path)
    done_ids = {r["source_image"] for r in results if "source_image" in r}

    print(f"Total images in filtered set: {len(image_paths)}")
    print(f"Already done: {len(done_ids)}")

    for img_path in image_paths:
        if img_path.name in done_ids:
            continue

        if not img_path.exists():
            rec = empty_record(img_path.name)
            rec["error"] = "image_not_found"
            results.append(rec)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            continue

        output = run_with_retry(img_path)

        if isinstance(output, dict) and "error" in output:
            rec = empty_record(img_path.name)
            rec["error"] = output["error"]
            results.append(rec)
        else:
            try:
                parsed = parse_llm_json(output)
                rec = empty_record(img_path.name)

                for key in ["item_name", "item_price", "sub_total", "tax", "total"]:
                    if key in parsed:
                        rec[key] = parsed[key]

                results.append(rec)

            except Exception as e:
                rec = empty_record(img_path.name)
                rec["error"] = f"parse_error: {e}"
                rec["raw_output"] = output
                results.append(rec)

        done_ids.add(img_path.name)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()