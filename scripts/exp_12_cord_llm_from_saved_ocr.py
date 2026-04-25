import os
import json
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct:fastest"
client = InferenceClient(api_key=HF_TOKEN)

PROMPT_TEMPLATE = """
You are extracting structured fields from OCR text of a receipt.

Return ONLY valid JSON with exactly these keys:
{{
  "item_name": null,
  "item_price": null,
  "sub_total": null,
  "tax": null,
  "total": null
}}

Rules:
- Copy values exactly as they appear in OCR text.
- Do NOT normalize commas, dots, spaces, dashes, or currency symbols.
- item_name:
  - choose the FIRST purchased item
  - do NOT choose store name, header, payment method, or summary line
- item_price:
  - choose the price associated with item_name
  - do NOT use subtotal, tax, total, cash, change, card payment, discount, or service values
- sub_total:
  - choose value near SUBTOTAL, SUB TOTAL, or Sub Total
- tax:
  - choose value near TAX, PB1, VAT, or service tax
- total:
  - choose the FINAL payable amount near TOTAL, GRAND TOTAL, TOTAL DUE, or AMOUNT DUE
- If missing or uncertain, return null.
- Do NOT wrap output in code fences.
- Return JSON only.

OCR TEXT:
{ocr_text}
""".strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def parse_llm_json(text: str):
    return json.loads(strip_code_fences(text))


def run_llm_once(prompt: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=220,
    )
    return completion.choices[0].message.content


def run_llm_with_retry(prompt: str, max_retries: int = 5):
    last_err = None
    for attempt in range(max_retries):
        try:
            return run_llm_once(prompt)
        except Exception as e:
            last_err = str(e)
            wait_s = 3 * (attempt + 1)
            print(f"  failed attempt {attempt+1}/{max_retries}: {last_err}")
            time.sleep(wait_s)
    raise RuntimeError(last_err or "unknown_llm_error")


def load_existing(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    results = load_existing(output_path)
    done_ids = {r["source_image"] for r in results if "source_image" in r}

    print(f"Already done: {len(done_ids)} / {len(ocr_data)}")

    for idx, rec in enumerate(ocr_data, start=1):
        source_image = rec["source_image"]

        if source_image in done_ids:
            continue

        ocr_text = rec.get("ocr_text", "")
        print(f"[{idx}/{len(ocr_data)}] {source_image}")

        out_row = {
            "source_image": source_image,
            "item_name": None,
            "item_price": None,
            "sub_total": None,
            "tax": None,
            "total": None,
        }

        if not ocr_text.strip():
            out_row["error"] = "empty_ocr_text"
        else:
            prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)
            try:
                raw = run_llm_with_retry(prompt, max_retries=5)
                parsed = parse_llm_json(raw)
                out_row.update(parsed)
            except Exception as e:
                out_row["error"] = str(e)

        results.append(out_row)
        done_ids.add(source_image)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()