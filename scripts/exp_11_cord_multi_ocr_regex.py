import json
import re
from pathlib import Path

# =========================
# PATHS
# =========================
OCR_FILES = {
    "paddle": Path("/root/lorex-ocr-llm-experiments/results/cord_ocr/paddle_ocr_cord_test_filtered.json"),
    "tesseract": Path("/root/lorex-ocr-llm-experiments/results/cord_ocr/tesseract_ocr_cord_test_filtered.json"),
    "easyocr": Path("/root/lorex-ocr-llm-experiments/results/cord_ocr/easyocr_ocr_cord_test_filtered.json"),
    "doctr": Path("/root/lorex-ocr-llm-experiments/results/cord_ocr/doctr_ocr_cord_test_filtered.json"),
    "openocr": Path("/root/lorex-ocr-llm-experiments/results/cord_ocr/openocr_ocr_cord_test_filtered.json"),
}

OUT_DIR = Path("/root/lorex-ocr-llm-experiments/results/cord_regex")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# HELPERS
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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

    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]

    # item_name
    for ln in lines:
        if re.search(r"\b(total|subtotal|tax|disc|cash|change|credit|debit|qty|amount|grand total)\b", ln, re.I):
            continue
        if re.fullmatch(r"[\d\s.,:/\-()$%]+", ln):
            continue
        if re.search(r"[A-Za-z]", ln):
            result["item_name"] = ln
            break

    # sub_total
    result["sub_total"] = first_match([
        r"\bsubtotal\b[^0-9\-]*([\d.,-]+)",
        r"\bsub total\b[^0-9\-]*([\d.,-]+)",
        r"\bsubtota[l1i]\b[^0-9\-]*([\d.,-]+)",
        r"\bsub tuta\b[^0-9\-]*([\d.,-]+)",
    ], text)

    # tax
    result["tax"] = first_match([
        r"\btax\b[^0-9\-]*([\d.,-]+)",
        r"\bpb1\b[^0-9\-]*([\d.,-]+)",
        r"\bvat\b[^0-9\-]*([\d.,-]+)",
        r"\bservice\s*tax\b[^0-9\-]*([\d.,-]+)",
        r"\btak\b[^0-9\-]*([\d.,-]+)",
    ], text)

    # total
    result["total"] = first_match([
        r"\bgrand total\b[^0-9\-]*([\d.,-]+)",
        r"\bamount due\b[^0-9\-]*([\d.,-]+)",
        r"\btotal sales\b[^0-9\-]*([\d.,-]+)",
        r"\btotal\b[^0-9\-]*([\d.,-]+)",
    ], text)

    # item_price
    price_candidates = []
    for ln in lines:
        if re.search(r"\b(subtotal|tax|total|disc|amount due|cash|change|grand total)\b", ln, re.I):
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

# =========================
# MAIN
# =========================
def main():
    for ocr_name, ocr_path in OCR_FILES.items():
        print(f"\nProcessing {ocr_name} ...")

        data = load_json(ocr_path)
        out_records = []

        for rec in data:
            source_image = rec["source_image"]
            text = rec.get("ocr_text", "")

            extracted = extract_with_regex(text)

            row = {
                "source_image": source_image,
                "ocr_text": text,
                **extracted,
            }

            if "error" in rec:
                row["ocr_error"] = rec["error"]

            out_records.append(row)

        out_path = OUT_DIR / f"{ocr_name}_regex_cord_test_filtered.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_records, f, ensure_ascii=False, indent=2)

        print(f"Saved: {out_path} | records: {len(out_records)}")

if __name__ == "__main__":
    main()