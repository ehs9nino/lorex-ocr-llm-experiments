import json
import re
from pathlib import Path

from paddleocr import PaddleOCR

IMG_DIR = Path("/root/lorex-ocr-llm-experiments/data/cord_test/images")
OUT_PATH = Path("/root/lorex-ocr-llm-experiments/results/paddle_regex_cord_test_15.json")
MAX_IMAGES = 15

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,
)

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def flatten_ocr_text(result) -> str:
    if not result:
        return ""

    page = result[0]

    if isinstance(page, dict) and "rec_texts" in page:
        texts = [t.strip() for t in page["rec_texts"] if isinstance(t, str) and t.strip()]
        return clean_text("\n".join(texts))

    return ""

def normalize_value(val):
    if val is None:
        return None
    val = str(val).strip()
    return val if val else None

def first_match(patterns, text, flags=re.IGNORECASE):
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(1).strip()
    return None

def extract_with_regex(text: str) -> dict:
    result = {
        "item_name": None,
        "item_price": None,
        "sub_total": None,
        "tax": None,
        "total": None,
    }

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # item_name: first non-summary text line with letters
    for ln in lines:
        if re.search(r"\b(total|subtotal|tax|disc|cash|change|credit|debit|qty|amount)\b", ln, re.I):
            continue
        if re.fullmatch(r"[\d\s.,:/\-()$]+", ln):
            continue
        if re.search(r"[A-Za-z]", ln):
            result["item_name"] = ln
            break

    # subtotal
    result["sub_total"] = first_match([
        r"\bsubtotal\b[^0-9\-]*([\d.,-]+)",
        r"\bsub total\b[^0-9\-]*([\d.,-]+)",
    ], text)

    # tax
    result["tax"] = first_match([
        r"\btax\b[^0-9\-]*([\d.,-]+)",
        r"\bpb1\b[^0-9\-]*([\d.,-]+)",
        r"\bservice\s*tax\b[^0-9\-]*([\d.,-]+)",
    ], text)

    # total
    result["total"] = first_match([
        r"\bgrand total\b[^0-9\-]*([\d.,-]+)",
        r"\btotal\b[^0-9\-]*([\d.,-]+)",
        r"\bamount due\b[^0-9\-]*([\d.,-]+)",
    ], text)

    # item_price: first money-like value before summary lines
    price_candidates = []
    for ln in lines:
        if re.search(r"\b(subtotal|tax|total|disc|amount due)\b", ln, re.I):
            continue
        found = re.findall(r"(?:Rp\.?\s*)?[\d]{1,3}(?:[.,][\d]{3})*(?:[.,]\d{2,3})?", ln)
        for f in found:
            if re.search(r"\d", f):
                price_candidates.append(f.strip())

    if price_candidates:
        result["item_price"] = price_candidates[-1]

    for k in result:
        result[k] = normalize_value(result[k])

    return result

def main():
    image_paths = sorted(IMG_DIR.glob("*.png"), key=lambda p: int(p.stem))[:MAX_IMAGES]
    print(f"Found {len(image_paths)} images")

    results = []

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Processing {img_path.name}")

        try:
            res = ocr.predict(str(img_path))
            raw_text = flatten_ocr_text(res)

            extracted = extract_with_regex(raw_text)

            rec = {
                "source_image": img_path.name,
                "ocr_text": raw_text,
                **extracted,
            }
            results.append(rec)

            print("--- OCR TEXT PREVIEW ---")
            print(raw_text[:500])
            print("------------------------")

        except Exception as e:
            print(f"ERROR on {img_path.name}: {e}")
            results.append({
                "source_image": img_path.name,
                "ocr_text": "",
                "item_name": None,
                "item_price": None,
                "sub_total": None,
                "tax": None,
                "total": None,
                "error": str(e),
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {OUT_PATH}")

if __name__ == "__main__":
    main()