
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path("experiments/noise_robustness")
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "load_number",
    "pickup_location",
    "pickup_time",
    "dropoff_location",
    "dropoff_time",
    "total_rate",
    "rate_per_mile",
]

METHODS = {
    "regex_tesseract": (
        "results/regex_tesseract_160.json",
        OUT / "regex_tesseract_mixed_135.json",
    ),
    "tesseract_llm_p1": (
        "results/tesseract_llm_160.json",
        OUT / "llm/tesseract_p1_mixed_135.json",
    ),
    "tesseract_llm_p2": (
        "results/tesseract_llm_160_prompt2.json",
        OUT / "llm/tesseract_p2_mixed_135.json",
    ),
    "paddle_llm_p2": (
        "results/paddle_llm_160_prompt2.json",
        OUT / "llm/paddle_p2_mixed_135.json",
    ),
    "easyocr_llm_p2": (
        "results/easyocr_llm_160_prompt2.json",
        OUT / "llm/easyocr_p2_mixed_135.json",
    ),
    "doctr_llm_p2": (
        "results/doctr_llm_160_prompt2.json",
        OUT / "llm/doctr_p2_mixed_135.json",
    ),
    "openocr_llm_p2": (
        "results/openocr_llm_160_prompt2.json",
        OUT / "llm/openocr_p2_mixed_135.json",
    ),
    "qwen_vlm": (
        "results/qwen_vlm_160_clean.json",
        OUT / "qwen_vlm_mixed_135.json",
    ),
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def key(row):
    # Match within the split and ignore extension differences.
    return (
        row["split"],
        Path(row["source_image"]).stem.lower(),
    )


def index_rows(rows, label):
    indexed = {}
    for row in rows:
        k = key(row)
        if k in indexed:
            raise ValueError(f"Duplicate in {label}: {k}")
        indexed[k] = row
    return indexed


def normalize(value):
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    if not text or text.lower() == "null":
        return None
    return text


def levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def score_pair(pred, gold):
    p = normalize(pred)
    g = normalize(gold)

    exact = float(p == g)

    if p is None and g is None:
        return exact, 1.0, 1.0, 0, 0, 0

    if p is None or g is None:
        tp = 0
        fp = int(p is not None)
        fn = int(g is not None)
        return exact, 0.0, 0.0, tp, fp, fn

    fuzzy = SequenceMatcher(
        None, p.lower(), g.lower()
    ).ratio()

    lev_sim = 1.0 - (
        levenshtein(p.lower(), g.lower())
        / max(len(p), len(g))
    )

    if p == g:
        tp, fp, fn = 1, 0, 0
    else:
        tp, fp, fn = 0, 1, 1

    return exact, fuzzy, lev_sim, tp, fp, fn


def evaluate(predictions, ground_truth, ids):
    per_field = {}

    for field in FIELDS:
        exact_sum = fuzzy_sum = lev_sum = 0.0
        tp = fp = fn = 0

        for image_id in ids:
            pred = predictions[image_id].get(field)
            gold = ground_truth[image_id].get(field)

            e, f, l, t, p, n = score_pair(pred, gold)

            exact_sum += e
            fuzzy_sum += f
            lev_sum += l
            tp += t
            fp += p
            fn += n

        count = len(ids)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_field[field] = {
            "exact_match": 100 * exact_sum / count,
            "fuzzy_score": 100 * fuzzy_sum / count,
            "lev_similarity": 100 * lev_sum / count,
            "precision": 100 * precision,
            "recall": 100 * recall,
            "f1": 100 * f1,
        }

    overall = {
        metric: sum(
            per_field[f][metric] for f in FIELDS
        ) / len(FIELDS)
        for metric in [
            "exact_match", "fuzzy_score",
            "lev_similarity", "precision",
            "recall", "f1"
        ]
    }

    return per_field, overall


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    gt_rows = []
    for split in ["train", "val", "test"]:
        for row in load(f"data/{split}.json"):
            gt_rows.append({"split": split, **row})

    gt = index_rows(gt_rows, "ground truth")

    # The original clean VLM subset defines the paired 135.
    subset = index_rows(
        load("results/qwen_vlm_160_clean.json"),
        "clean VLM subset",
    )
    ids = sorted(subset, key=lambda k: (k[0], int(k[1])))

    assert len(ids) == 135
    assert set(ids).issubset(gt)

    summary = []
    detailed = []

    for method, (clean_path, noisy_path) in METHODS.items():
        clean = index_rows(load(clean_path), f"{method} clean")
        noisy = index_rows(load(noisy_path), f"{method} noisy")

        for label, records in [("clean", clean), ("noisy", noisy)]:
            missing = set(ids) - set(records)
            if missing:
                raise ValueError(
                    f"{method} {label}: {len(missing)} missing IDs; "
                    f"examples: {list(missing)[:3]}"
                )

        # Never silently exclude failed predictions.
        failed = [
            k for k in ids
            if noisy[k].get("_status") == "failed"
        ]
        if failed:
            print(f"WARNING: {method} has {len(failed)} failed calls")

        clean_fields, clean_overall = evaluate(clean, gt, ids)
        noisy_fields, noisy_overall = evaluate(noisy, gt, ids)

        summary.append({
            "method": method,
            "n": len(ids),
            **{
                f"clean_{metric}": round(clean_overall[metric], 2)
                for metric in clean_overall
            },
            **{
                f"noisy_{metric}": round(noisy_overall[metric], 2)
                for metric in noisy_overall
            },
            **{
                f"delta_{metric}_pp": round(
                    noisy_overall[metric] - clean_overall[metric], 2
                )
                for metric in clean_overall
            },
        })

        for field in FIELDS:
            detailed.append({
                "method": method,
                "field": field,
                **{
                    f"clean_{m}": round(clean_fields[field][m], 2)
                    for m in clean_fields[field]
                },
                **{
                    f"noisy_{m}": round(noisy_fields[field][m], 2)
                    for m in noisy_fields[field]
                },
                **{
                    f"delta_{m}_pp": round(
                        noisy_fields[field][m]
                        - clean_fields[field][m], 2
                    )
                    for m in clean_fields[field]
                },
            })

        print(
            f"{method:20s} "
            f"EM {clean_overall['exact_match']:6.2f} -> "
            f"{noisy_overall['exact_match']:6.2f} | "
            f"F1 {clean_overall['f1']:6.2f} -> "
            f"{noisy_overall['f1']:6.2f}"
        )

    write_csv(OUT / "robustness_summary.csv", summary)
    write_csv(OUT / "robustness_per_field.csv", detailed)

    print("\nSaved:")
    print(OUT / "robustness_summary.csv")
    print(OUT / "robustness_per_field.csv")


if __name__ == "__main__":
    main()
