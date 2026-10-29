
import json
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path("experiments/noise_robustness")
MANIFEST = ROOT / "common_subset.json"
OUTPUT = ROOT / "images/mixed"

OUTPUT.mkdir(parents=True, exist_ok=True)

with open(MANIFEST, encoding="utf-8") as f:
    records = json.load(f)

# Reproducible noise assignments
seed = 42
rng = random.Random(seed)
np_rng = np.random.default_rng(seed)

noise_types = ["blur", "gaussian_noise", "jpeg", "rotation"]

# Balanced assignment, then shuffle
assignments = [
    noise_types[i % len(noise_types)]
    for i in range(len(records))
]
rng.shuffle(assignments)

metadata = []

for record, noise_type in zip(records, assignments):
    source = Path(record["source_path"])
    image = Image.open(source).convert("RGB")
    params = {}

    if noise_type == "blur":
        params = {"radius": 0.8}
        result = image.filter(
            ImageFilter.GaussianBlur(radius=0.8)
        )

    elif noise_type == "gaussian_noise":
        params = {"sigma": 15}
        arr = np.asarray(image, dtype=np.float32)
        noise = np_rng.normal(0, 15, arr.shape)
        result = Image.fromarray(
            np.clip(arr + noise, 0, 255).astype(np.uint8)
        )

    elif noise_type == "jpeg":
        params = {"quality": 35}
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=35)
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")

    elif noise_type == "rotation":
        angle = rng.choice([-3, 3])
        params = {"angle_degrees": angle}
        result = image.rotate(
            angle,
            expand=True,
            fillcolor="white"
        )

    # Preserve image ID and original split
    split_dir = OUTPUT / record["split"]
    split_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(record["source_image"]).stem + ".png"
    output_path = split_dir / filename
    result.save(output_path)

    metadata.append({
        "split": record["split"],
        "source_image": record["source_image"],
        "original_path": str(source),
        "noisy_path": str(output_path),
        "noise_type": noise_type,
        "parameters": params,
    })

with open(ROOT / "noise_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

from collections import Counter

print("Generated:", len(metadata))
print("Noise distribution:", dict(Counter(
    item["noise_type"] for item in metadata
)))
print("Metadata saved:", ROOT / "noise_metadata.json")
