import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from paddleocr import PaddleOCR

# =========================
# CONFIG
# =========================
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

IMG_DIR = Path("/root/lorex-ocr-llm-experiments/data/cord_test/images")
OUT_PATH = Path("/root/lorex-ocr-llm-experiments/results/paddle_llm_cord_test_15.json")
MAX_IMAGES = 15
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto",
)

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,
)

PROMPT_TEMPLATE = """
Extract structured fields from this receipt OCR text.

Return ONLY valid JSON with exactly these keys:
{{
  "item_name": null,
  "item_price": null,
  "sub_total": null,
  "tax": null,
  "total": null
}}

Rules:
- Copy values exactly as they appear in the OCR text.
- Do not normalize currency, commas, dots, spaces, or symbols.
- item_name = the first main purchased item, not store name, not payment method, not summary line.
- item_price = the price associated with item_name, not subtotal, tax, cash, or change.
- sub_total = value near keywords like SUBTOTAL, SUB TOTAL, or Sub Total.
- tax = value near keywords like TAX, PB1, or service tax.
- total = final payable amount near keywords like TOTAL, GRAND TOTAL, or Total.
- Do NOT use CASH, CHANGE, CARD PAYMENT, CREDIT CARD, or DISCOUNT as total.
- If multiple totals appear, choose the final payable amount.
- If unsure, output null.
- Return JSON only.

OCR TEXT:
{ocr_text}
""".strip()


# =========================
# HELPERS
# =========================
def clean_text(text: str) -> str:
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])


def flatten_ocr_text(result) -> str:
    if not result:
        return ""

    page = result[0]

    if isinstance(page, dict) and "rec_texts" in page:
        texts = [t.strip() for t in page["rec_texts"] if isinstance(t, str) and t.strip()]
        return clean_text("\n".join(texts))

    return ""


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


def empty_record(image_name: str, ocr_text: str = "") -> dict:
    return {
        "source_image": image_name,
        "ocr_text": ocr_text,
        "item_name": None,
        "item_price": None,
        "sub_total": None,
        "tax": None,
        "total": None,
    }


def run_llm(ocr_text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300,
    )

    return completion.choices[0].message.content


def run_llm_with_retry(ocr_text: str, max_retries: int = 4):
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"  LLM attempt {attempt+1}")
            return run_llm(ocr_text)
        except Exception as e:
            last_error = str(e)
            print(f"  LLM failed: {e}")
            time.sleep(3 * (attempt + 1))

    return {"error": last_error or "failed_after_retries"}


# =========================
# MAIN
# =========================
def main():
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")

    image_paths = sorted(IMG_DIR.glob("*.png"), key=lambda p: int(p.stem))[:MAX_IMAGES]
    print(f"Found {len(image_paths)} images")

    results = []

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Processing {img_path.name}")

        try:
            # OCR
            ocr_result = ocr.predict(str(img_path))
            ocr_text = flatten_ocr_text(ocr_result)

            print("--- OCR TEXT PREVIEW ---")
            print(ocr_text[:500])
            print("------------------------")

            # LLM
            llm_out = run_llm_with_retry(ocr_text)

            if isinstance(llm_out, dict) and "error" in llm_out:
                rec = empty_record(img_path.name, ocr_text)
                rec["error"] = llm_out["error"]
                results.append(rec)
                continue

            try:
                parsed = parse_llm_json(llm_out)
                rec = empty_record(img_path.name, ocr_text)

                for key in ["item_name", "item_price", "sub_total", "tax", "total"]:
                    if key in parsed:
                        rec[key] = parsed[key]

                results.append(rec)

            except Exception as e:
                rec = empty_record(img_path.name, ocr_text)
                rec["error"] = f"parse_error: {e}"
                rec["raw_output"] = llm_out
                results.append(rec)

        except Exception as e:
            rec = empty_record(img_path.name, "")
            rec["error"] = str(e)
            results.append(rec)

        # save progress after each image
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(1)

    print(f"\nSaved results to: {OUT_PATH}")


if __name__ == "__main__":
    main()