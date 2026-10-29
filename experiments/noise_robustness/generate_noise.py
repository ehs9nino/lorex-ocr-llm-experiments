
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import io

# Original LoREx-160 test images
INPUT_DIR = Path("../LoREx-160/rate_confirmations/images/test")

# Output folder (relative to project root)
OUTPUT_DIR = Path("experiments/noise_robustness/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

def gaussian_noise(image, sigma=15):
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = rng.normal(0, sigma, array.shape)
    result = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(result)

def jpeg_compression(image, quality=35):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

images = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

# Pilot: first image only
for path in images:
    original = Image.open(path).convert("RGB")

    variants = {
        "blur": original.filter(ImageFilter.GaussianBlur(radius=1.5)),
        "gaussian_noise": gaussian_noise(original),
        "jpeg": jpeg_compression(original),
    }

    for noise_type, image in variants.items():
        folder = OUTPUT_DIR / noise_type
        folder.mkdir(parents=True, exist_ok=True)
        image.save(folder / f"{path.stem}.png")

        print(f"Saved: {folder / (path.stem + '.png')}")

print("Noise generation completed.")
