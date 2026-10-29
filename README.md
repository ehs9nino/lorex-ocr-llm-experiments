# LoREx-160: OCR, LLM, and Vision–Language Document Extraction Benchmark

Experimental code, evaluation notebooks, model predictions, and benchmark results accompanying the research manuscript:

**LoREx-160: A Controlled Benchmark of Rule-Based, OCR–LLM, and Vision–Language Pipelines for Logistics Document Information Extraction**

**Authors:** Ehsan Qader, Dmitry Efremenko, and Denis Derkach

**Manuscript status:** Under review (2026)

## Overview

LoREx-160 is a research benchmark for structured information extraction from logistics documents.

The study systematically compares three extraction approaches:

1. **OCR + Regex:** Deterministic extraction using OCR and regular expressions.
2. **OCR + LLM:** Text recognition followed by structured extraction using Qwen2.5-7B-Instruct.
3. **Direct VLM:** End-to-end extraction from document images using Qwen2.5-VL.

The experiments investigate how recognition quality, semantic field assignment, prompt design, and image degradation affect extraction performance.

The study evaluates pretrained models; no model fine-tuning is performed.

## Datasets

### LoREx-160

LoREx-160 contains **160 manually annotated logistics-document images**, primarily digital screenshots of rate-confirmation and dispatch workflows.

Each document is annotated using seven target fields:

| Field | Description |
|---|---|
| `load_number` | Shipment or load identifier |
| `pickup_location` | Pickup facility or address |
| `pickup_time` | Scheduled pickup date and time |
| `dropoff_location` | Delivery facility or address |
| `dropoff_time` | Scheduled delivery date and time |
| `total_rate` | Total shipment payment |
| `rate_per_mile` | Payment per mile |

The annotation files are organized into predefined splits:

| Split | Documents |
|---|---:|
| Training | 120 |
| Validation | 16 |
| Test | 24 |
| **Total** | **160** |

**Important:** The main comparative evaluation uses a common matched subset of **135 documents**, rather than exclusively the 24-document test split.

Dataset repository: [LoREx-160](https://github.com/ehs9nino/LoREx-160)

### CORD

The CORD receipt dataset is used for supplementary cross-domain evaluation.

Dataset: [CORD v2 on Hugging Face](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

## Extraction Methods

### OCR + Regex

OCR-recognized text is processed using deterministic patterns and rules to identify the target fields.

### OCR + LLM

The following OCR engines are evaluated:

- Tesseract
- PaddleOCR
- EasyOCR
- docTR
- OpenOCR

OCR text is passed to **Qwen2.5-7B-Instruct**, which generates structured field predictions using task-specific prompts.

### Direct Vision–Language Model

**Qwen2.5-VL** processes document images directly, without requiring an external OCR stage.

## Experiments

### 1. Matched Clean Evaluation

Eight extraction configurations are compared using the same matched subset of 135 logistics documents.

The evaluation examines overall performance and individual field extraction accuracy.

The aligned comparison retains 21 empty Qwen2.5-VL predictions associated with unsuccessful inference, rather than excluding these records.

Notebook: `notebooks/LoREx_clean_benchmark.ipynb`

### 2. Noise Robustness

The robustness experiments investigate how synthetic image degradations influence extraction accuracy.

They compare performance on clean and degraded document images and analyze differences across extraction methods and target fields.

Notebooks:

- `notebooks/LoREx_noise_robustness_evaluation.ipynb`
- `notebooks/LoREx_noise_robustness_comparison.ipynb`

### 3. Controlled Prompt Ablation

Controlled experiments investigate the contribution of domain-specific prompt instructions and JSON response recovery.

The experiment uses fixed OCR inputs and Qwen2.5-7B-Instruct to isolate the effects of prompt changes.

The study includes comparisons of field-specific guidance and structured-output handling.

Notebook: `notebooks/LoREx_prompt_ablation.ipynb`

### 4. CORD Benchmark

Supplementary experiments examine the behavior of extraction methods on receipt documents outside the logistics domain.

Notebooks:

- `notebooks/CORD_benchmark.ipynb`
- `notebooks/CORD_prompt_postprocessing_pilot.ipynb`

## Repository Structure

```text
lorex-ocr-llm-experiments/
├── data/
│   ├── train.json
│   ├── val.json
│   └── test.json
│
├── experiments/
│   ├── matched_inference/
│   ├── noise_robustness/
│   └── prompt_ablation/
│
├── notebooks/
│   ├── LoREx_clean_benchmark.ipynb
│   ├── LoREx_noise_robustness_evaluation.ipynb
│   ├── LoREx_noise_robustness_comparison.ipynb
│   ├── LoREx_prompt_ablation.ipynb
│   ├── CORD_benchmark.ipynb
│   ├── CORD_prompt_postprocessing_pilot.ipynb
│   └── setup_cord.ipynb
│
├── results/
├── scripts/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The `experiments/` directory contains additional experiment scripts, saved predictions, evaluation tables, and selected visualizations.

## Evaluation Metrics

The study uses four complementary measures:

| Metric | Purpose |
|---|---|
| Exact Match (EM) | Measures strict agreement between predicted and reference field values |
| Field-level F1 | Combines extraction precision and recall |
| Fuzzy Similarity | Measures partial string agreement |
| Normalized Levenshtein Similarity | Measures character-level similarity using edit distance |

These metrics distinguish exact structured extraction accuracy from partial textual similarity.

The experiments also report field-specific results and comparative visualizations.

**Evaluation note:** Empty reference fields and unsuccessful model predictions require consistent handling. The implementation and evaluation notebooks should be consulted for the precise scoring rules.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ehs9nino/lorex-ocr-llm-experiments.git
cd lorex-ocr-llm-experiments
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Some OCR engines may require additional system packages, model downloads, or environment-specific configuration.

### 4. Configure API access

Certain experiments use models accessed through Hugging Face.

Create a local `.env` file when required:

```env
HF_TOKEN=your_huggingface_token
```

Do not commit API keys or credentials.

## Usage

### Run the OCR + Regex baseline

```bash
python scripts/exp_01_lorex_regex_tesseract.py
```

This script is the original LoREx OCR + Regex experiment entry point.

### Explore the evaluation notebooks

```bash
jupyter notebook
```

Open the relevant notebook in `notebooks/`.

Some notebooks analyze previously generated predictions and therefore require their associated result files. Full inference reproduction may additionally require dataset images, model access, and experiment-specific configuration.

## Models and OCR Engines

**Language model:** [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

**Vision–language model:** [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

**OCR engines:**

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [docTR](https://github.com/mindee/doctr)
- [OpenOCR](https://github.com/Topdu/OpenOCR)

## Research Scope and Limitations

LoREx-160 primarily contains digital logistics-document screenshots, rather than a comprehensive collection of naturally scanned or photographed documents.

The benchmark is limited to the evaluated model configurations and datasets. Results should not be interpreted as universal rankings across all OCR, LLM, and VLM systems.

The experiments are intended to support transparent comparisons and further research into document information extraction.

## Citation

If you use the LoREx-160 dataset or the experimental resources provided in this repository, please cite the associated dataset and research manuscript.

### Dataset

**LoREx-160 Dataset (2026).** Zenodo.

- **DOI:** https://doi.org/10.5281/zenodo.18687787
- **Dataset:** https://zenodo.org/records/18687787
- **Experiments:** https://github.com/ehs9nino/lorex-ocr-llm-experiments

> For the exact dataset authors and title, please use the citation metadata provided by Zenodo.

### Research Manuscript

Qader, E., Efremenko, D., & Derkach, D. (2026).
*LoREx-160: A Controlled Benchmark of Rule-Based, OCR–LLM, and Vision–Language Pipelines for Logistics Document Information Extraction.*

**Status:** Manuscript under review.

The final bibliographic citation will be updated upon publication.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.