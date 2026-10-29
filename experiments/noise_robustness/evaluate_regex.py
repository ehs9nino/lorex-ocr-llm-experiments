
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd

ROOT = Path("experiments/noise_robustness")

FIELDS = [
    "load_number",
    "pickup_location",
    "pickup_time",
    "dropoff_location",
    "dropoff_time",
    "total_rate",
    "rate_per_mile",
]

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def image_id(name):
    return Path(str(name)).stem

def normalize(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "null":
        return None
    return re.sub(r"\s+", " ", x)

def to_dict(rows):
    result = {}
    for row in rows:
        key = image_id(row["source_image"])
        if key in result:
            raise ValueError(f"Duplicate ID: {key}")
        result[key] = {
            f: normalize(row.get(f)) for f in FIELDS
        }
    return result

def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))

def fuzzy_score(pred, gold):
    pred, gold = normalize(pred), normalize(gold)
    if pred is None and gold is None:
        return 1.0
    if pred is None or gold is None:
        return 0.0
    return SequenceMatcher(
        None, pred.lower(), gold.lower()
    ).ratio()

def field_confusion(pred, gold):
    pred, gold = normalize(pred), normalize(gold)
    if gold is None and pred is None:
        return 0, 0, 0
    if gold is not None and pred is not None and pred == gold:
        return 1, 0, 0
    if gold is None:
        return 0, 1, 0
    if pred is None:
        return 0, 0, 1
    return 0, 1, 1

def levenshtein_distance(a, b):
    a, b = normalize(a), normalize(b)
    if a is None and b is None:
        return 0
    if a is None:
        return len(b)
    if b is None:
        return len(a)
    a, b = a.lower(), b.lower()
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(
                new[j-1] + 1,
                dp[j] + 1,
                dp[j-1] + (ca != cb)
            ))
        dp = new
    return dp[-1]

def normalized_levenshtein(pred, gold):
    pred, gold = normalize(pred), normalize(gold)
    if pred is None and gold is None:
        return 0.0
    if pred is None or gold is None:
        return 1.0
    return levenshtein_distance(pred, gold) / max(
        len(pred), len(gold)
    )

def evaluate(predictions, ground_truth, ids):
    rows = []
    for field in FIELDS:
        exacts, fuzzies, distances = [], [], []
        tp = fp = fn = 0

        for key in ids:
            pred = predictions[key][field]
            gold = ground_truth[key][field]

            exacts.append(exact_match(pred, gold))
            fuzzies.append(fuzzy_score(pred, gold))
            distances.append(normalized_levenshtein(pred, gold))

            a, b, c = field_confusion(pred, gold)
            tp += a
            fp += b
            fn += c

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        rows.append({
            "field": field,
            "exact_match": sum(exacts) / len(exacts),
            "fuzzy_score": sum(fuzzies) / len(fuzzies),
            "lev_similarity": 1 - sum(distances) / len(distances),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

    return pd.DataFrame(rows)

# Ground truth
gt_rows = []
for split in ("train", "val", "test"):
    gt_rows.extend(load(f"data/{split}.json"))

gt = to_dict(gt_rows)

# Same 135 IDs as Figure 14
manifest = load(ROOT / "common_subset.json")
ids = [image_id(row["source_image"]) for row in manifest]

assert len(ids) == len(set(ids)) == 135

clean = to_dict(load("results/regex_tesseract_160.json"))
noisy = to_dict(load(
    ROOT / "results/regex_tesseract_mixed_135.json"
))

for name, records in [("GT", gt), ("Clean", clean), ("Noisy", noisy)]:
    missing = set(ids) - set(records)
    if missing:
        raise ValueError(f"{name} missing IDs: {sorted(missing)}")

clean_eval = evaluate(clean, gt, ids)
noisy_eval = evaluate(noisy, gt, ids)

comparison = pd.DataFrame({
    "field": FIELDS,
    "clean_exact_match": clean_eval["exact_match"] * 100,
    "noisy_exact_match": noisy_eval["exact_match"] * 100,
    "delta_pp": (
        noisy_eval["exact_match"] - clean_eval["exact_match"]
    ) * 100,
    "clean_f1": clean_eval["f1_score"] * 100,
    "noisy_f1": noisy_eval["f1_score"] * 100,
})

comparison.loc[len(comparison)] = {
    "field": "OVERALL (macro)",
    "clean_exact_match": comparison["clean_exact_match"].mean(),
    "noisy_exact_match": comparison["noisy_exact_match"].mean(),
    "delta_pp": comparison["delta_pp"].mean(),
    "clean_f1": comparison["clean_f1"].mean(),
    "noisy_f1": comparison["noisy_f1"].mean(),
}

output = ROOT / "results/regex_clean_vs_noisy.csv"
comparison.to_csv(output, index=False)

print("\nTesseract + Regex: Clean vs Mixed Noise")
print(comparison.round(2).to_string(index=False))
print("\nSaved:", output)
