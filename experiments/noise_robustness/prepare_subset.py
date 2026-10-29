
import json
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = ROOT / "experiments/noise_robustness"
IMAGES = ROOT.parent / "LoREx-160/rate_confirmations/images"

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def image_id(name):
    return Path(str(name)).stem

# Reproduce the selection in exp_7_lorex_evaluation_combined.ipynb
gt = []
for split in ("train", "val", "test"):
    for row in load(DATA / f"{split}.json"):
        gt.append({**row, "_split": split})

qwen = load(RESULTS / "qwen_vlm_160_clean.json")
qwen_ids = {image_id(r["source_image"]) for r in qwen}
gt_ids = {image_id(r["source_image"]) for r in gt}

common_ids = qwen_ids & gt_ids

# Ensure all three main configurations have matching records
for filename in (
    "paddle_llm_160_prompt2.json",
    "regex_tesseract_160.json",
):
    rows = load(RESULTS / filename)
    ids = {image_id(r["source_image"]) for r in rows}
    missing = common_ids - ids
    if missing:
        raise ValueError(f"{filename} missing IDs: {sorted(missing)}")

selected = [
    r for r in gt
    if image_id(r["source_image"]) in common_ids
]
selected.sort(key=lambda r: int(image_id(r["source_image"])))

manifest = []
for row in selected:

    split = row["_split"]
    name = row["source_image"]

    # Find the actual image, regardless of its extension
    split_dir = IMAGES / split
    stem = Path(name).stem

    matches = [
        p for p in split_dir.iterdir()
        if p.is_file()
        and p.stem == stem
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one image for {split}/{stem}, "
            f"found {len(matches)}: {matches}"
        )

    path = matches[0]


    manifest.append({
        "split": split,
        "source_image": name,
        "source_path": str(path),
    })

OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "common_subset.json"

with open(path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Qwen unique IDs:", len(qwen_ids))
print("Ground-truth matched IDs:", len(common_ids))
print("Selected images:", len(manifest))
print("Saved:", path)
print("First five:", manifest[:5])
