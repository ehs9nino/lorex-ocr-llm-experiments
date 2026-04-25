import os
import json
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto",
)

sample_names = [f"{i}.jpg" for i in range(1, 16)]
def get_url(image_name: str) -> str:
    return f"https://raw.githubusercontent.com/ehs9nino/LoREx-160/main/rate_confirmations/images/val/{image_name}"

def run_one(image_name: str) -> str:
    image_url = get_url(image_name)
    
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
"""

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

def run_with_retry(image_name: str, max_retries: int = 5):
    last_error = None
    for attempt in range(max_retries):
        try:
            print(f"[{image_name}] attempt {attempt+1}")
            return run_one(image_name)
        except Exception as e:
            last_error = str(e)
            print(f"[{image_name}] failed: {e}")
            time.sleep(2 * (attempt + 1))
    return {"error": last_error or "failed_after_retries"}

def main():
    results = []

    for image_name in sample_names:
        output = run_with_retry(image_name)

        if isinstance(output, dict) and "error" in output:
            results.append({
                "source_image": image_name,
                "error": output["error"]
            })
        else:
            results.append({
                "source_image": image_name,
                "raw_output": output
            })

        time.sleep(2)

    os.makedirs("results", exist_ok=True)
    out_path = "results/qwen_vlm_raw_15.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()