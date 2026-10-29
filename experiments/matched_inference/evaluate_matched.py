import csv
import importlib.util
from pathlib import Path

BASE = Path("experiments/noise_robustness/evaluate_all_methods.py")
ROOT = Path("experiments/matched_inference")
OUT = ROOT / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("original_evaluator", BASE)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

FIELDS = ev.FIELDS
METRICS = [
    "exact_match", "fuzzy_score", "lev_similarity",
    "precision", "recall", "f1",
]

NOISE = Path("experiments/noise_robustness/results")
CLEAN = ROOT / "predictions/clean"

METHODS = {
    "regex_tesseract": (
        "results/regex_tesseract_160.json",
        CLEAN / "regex_tesseract_clean_135.json",
        NOISE / "regex_tesseract_mixed_135.json",
    ),
    "tesseract_llm_p1": (
        "results/tesseract_llm_160.json",
        CLEAN / "tesseract_p1_clean_135.json",
        NOISE / "llm/tesseract_p1_mixed_135.json",
    ),
    "tesseract_llm_p2": (
        "results/tesseract_llm_160_prompt2.json",
        CLEAN / "tesseract_p2_clean_135.json",
        NOISE / "llm/tesseract_p2_mixed_135.json",
    ),
    "paddle_llm_p2": (
        "results/paddle_llm_160_prompt2.json",
        CLEAN / "paddle_p2_clean_135.json",
        NOISE / "llm/paddle_p2_mixed_135.json",
    ),
    "easyocr_llm_p2": (
        "results/easyocr_llm_160_prompt2.json",
        CLEAN / "easyocr_p2_clean_135.json",
        NOISE / "llm/easyocr_p2_mixed_135.json",
    ),
    "doctr_llm_p2": (
        "results/doctr_llm_160_prompt2.json",
        CLEAN / "doctr_p2_clean_135.json",
        NOISE / "llm/doctr_p2_mixed_135.json",
    ),
    "openocr_llm_p2": (
        "results/openocr_llm_160_prompt2.json",
        CLEAN / "openocr_p2_clean_135.json",
        NOISE / "llm/openocr_p2_mixed_135.json",
    ),
    "qwen_vlm": (
        "results/qwen_vlm_160_clean.json",
        NOISE / "qwen_vlm_clean_rerun_135.json",
        NOISE / "qwen_vlm_mixed_135.json",
    ),
}

gt_rows = []
for split in ("train", "val", "test"):
    gt_rows.extend(
        {"split": split, **row}
        for row in ev.load(f"data/{split}.json")
    )

gt = ev.index_rows(gt_rows, "ground truth")

subset = ev.index_rows(
    ev.load("results/qwen_vlm_160_clean.json"),
    "original VLM subset",
)

ids = sorted(subset)
assert len(ids) == 135
assert set(ids).issubset(gt)

summary = []
details = []

for method, paths in METHODS.items():
    conditions = {}
    field_scores = {}

    for condition, path in zip(
        ("historical_clean", "matched_clean", "noisy"),
        paths,
    ):
        rows = ev.index_rows(ev.load(path), f"{method}/{condition}")

        missing = set(ids) - set(rows)
        if missing:
            raise ValueError(
                f"{method}/{condition}: {len(missing)} missing IDs"
            )

        failed = [
            k for k in ids
            if rows[k].get("_status") == "failed"
        ]
        if failed:
            raise ValueError(
                f"{method}/{condition}: {len(failed)} failed predictions"
            )

        fields, overall = ev.evaluate(rows, gt, ids)
        conditions[condition] = overall
        field_scores[condition] = fields

    result = {"method": method, "n": len(ids)}

    for condition, scores in conditions.items():
        for metric in METRICS:
            result[f"{condition}_{metric}"] = round(
                scores[metric], 2
            )

    for metric in METRICS:
        result[f"delta_noisy_vs_matched_{metric}_pp"] = round(
            conditions["noisy"][metric]
            - conditions["matched_clean"][metric],
            2,
        )
        result[f"delta_noisy_vs_historical_{metric}_pp"] = round(
            conditions["noisy"][metric]
            - conditions["historical_clean"][metric],
            2,
        )

    summary.append(result)

    for field in FIELDS:
        item = {"method": method, "field": field}

        for condition in (
            "historical_clean", "matched_clean", "noisy"
        ):
            for metric in METRICS:
                item[f"{condition}_{metric}"] = round(
                    field_scores[condition][field][metric], 2
                )

        for metric in METRICS:
            item[f"delta_noisy_vs_matched_{metric}_pp"] = round(
                field_scores["noisy"][field][metric]
                - field_scores["matched_clean"][field][metric],
                2,
            )

        details.append(item)

    print(
        f"{method:20s} "
        f"historical={conditions['historical_clean']['f1']:6.2f} "
        f"matched={conditions['matched_clean']['f1']:6.2f} "
        f"noisy={conditions['noisy']['f1']:6.2f} "
        f"delta={result['delta_noisy_vs_matched_f1_pp']:+6.2f} pp",
        flush=True,
    )

ev.write_csv(OUT / "matched_summary.csv", summary)
ev.write_csv(OUT / "matched_per_field.csv", details)

print("\nSaved:")
print(OUT / "matched_summary.csv")
print(OUT / "matched_per_field.csv")
