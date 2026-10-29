
import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

ROOT = Path("experiments/noise_robustness")
OCR_DIR = ROOT / "results/ocr"
OUT_DIR = ROOT / "results/llm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "load_number", "pickup_location", "pickup_time",
    "dropoff_location", "dropoff_time",
    "total_rate", "rate_per_mile"
]

METHODS = {
    "tesseract_p1": (
        "tesseract", "exp_03_lorex_tesseract_llm_prompt1.py", False
    ),
    "tesseract_p2": (
        "tesseract", "exp_03_lorex_tesseract_llm_prompt2.py", True
    ),
    "paddle_p2": (
        "paddle", "exp_04_lorex_paddle_llm_prompt2.py", True
    ),
    "easyocr_p2": (
        "easyocr", "exp_05_lorex_easyocr_llm_prompt2.py", True
    ),
    "doctr_p2": (
        "doctr", "exp_06_lorex_doctr_llm_prompt2.py", True
    ),
    "openocr_p2": (
        "openocr", "exp_08_lorex_openocr_llm_prompt2.py", True
    ),
}

def load_original_prompt(filename):
    """Extract the original build_prompt without running the script."""
    source = (Path("scripts") / filename).read_text(
        encoding="utf-8"
    )

    import ast
    tree = ast.parse(source)

    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_prompt"
    ]
    if len(functions) != 1:
        raise ValueError(f"Cannot find build_prompt in {filename}")

    # Compile only the prompt function, not module-level OCR/API code.
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {}
    exec(compile(module, filename, "exec"), namespace)
    return namespace["build_prompt"]

def parse_response(text):
    """Original P2 JSON parsing behavior."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object")
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data

def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=METHODS, required=True
    )
    parser.add_argument(
        "--limit", type=int, default=135
    )
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in .env")

    engine, script, use_temp = METHODS[args.method]
    build_prompt = load_original_prompt(script)

    client = InferenceClient(
        api_key=token,
        provider="featherless-ai"
    )

    ocr_path = OCR_DIR / f"{engine}_mixed_135.json"
    ocr_rows = json.loads(
        ocr_path.read_text(encoding="utf-8")
    )

    if len(ocr_rows) != 135:
        raise ValueError("Expected 135 OCR records")

    output = OUT_DIR / f"{args.method}_mixed_135.json"
    if output.exists():
        results = json.loads(
            output.read_text(encoding="utf-8")
        )
    else:
        results = []

    completed = {
        (r["split"], r["source_image"])
        for r in results
        if r.get("_status") == "ok"
    }

    for index, row in enumerate(
        ocr_rows[:args.limit], start=1
    ):
        key = (row["split"], row["source_image"])

        if key in completed:
            continue

        prompt = build_prompt(
            row["source_image"],
            row["ocr_text"]
        )

        result = {
            "split": row["split"],
            "source_image": row["source_image"],
            **{f: None for f in FIELDS},
        }

        error = None
        for attempt in range(1, 4):
            try:
                kwargs = {
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 300,
                }
                if use_temp:
                    kwargs["temperature"] = 0

                completion = client.chat.completions.create(
                    **kwargs
                )

                raw = completion.choices[0].message.content
                data = parse_response(raw)

                for field in FIELDS:
                    result[field] = data.get(field)

                result["_status"] = "ok"
                error = None
                break

            except Exception as exc:
                error = str(exc)
                print(
                    f"{args.method} {index}: "
                    f"attempt {attempt} failed: {error}",
                    flush=True
                )
                if attempt < 3:
                    time.sleep(3 * attempt)

        if error:
            result["_status"] = "failed"
            result["_error"] = error

        # Replace an earlier failed record rather than duplicating it.
        results = [
            r for r in results
            if (r["split"], r["source_image"]) != key
        ]
        results.append(result)
        save(output, results)

        print(
            f"[{args.method}] {index}/{args.limit} "
            f"{row['source_image']} "
            f"{result['_status']}",
            flush=True
        )

    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
