# LoREx OCR–LLM Experiments

This repository contains the experimental code and evaluation pipelines for our study on structured information extraction from noisy documents using OCR and Large Language Models (LLMs).

## Overview

We benchmark multiple document processing pipelines:
- OCR + Regex
- OCR + LLM
- Vision-Language Models (VLMs)

The evaluation is conducted on:
- LoREx-160 dataset (rate confirmations)
- CORD dataset (receipts)

The goal is to analyze how OCR quality, prompt design, and model choice affect structured information extraction.

---

## Repository Structure

data/           # Datasets (CORD subset, ground truth JSON)  
notebooks/      # Evaluation and visualization notebooks  
results/        # Model outputs and experiment results  
scripts/        # OCR + LLM pipeline scripts  

---

## Datasets

- LoREx-160 (Rate Confirmations):  
  https://github.com/ehs9nino/LoREx-160  

- CORD (Receipt dataset):  
  https://huggingface.co/datasets/naver-clova-ix/cord-v2  

---

## Setup


### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```


---

## Usage

Run experiments:

python scripts/exp_01_regex_tesseract_all160.py

Or open notebooks:

jupyter notebook

---

## Evaluation

We evaluate extraction quality using:

- Exact Match  
- F1 Score  
- Fuzzy Matching  
- Normalized Levenshtein Distance  

Results are visualized using heatmaps and comparison plots.

---

## Key Idea

Modern OCR systems produce high-quality text, but they do not perform semantic field extraction.  
This repository demonstrates how LLMs interpret OCR outputs and convert them into structured JSON representations.

---

## Paper

This repository accompanies the paper:

"Traffic Document Processing with Large Language Models: A Benchmark for Information Extraction from Noisy OCR"

---

### Notes

## Notes

- Experiments are conducted on subsets of datasets for reproducibility  
- Some models require API access via Hugging Face  

### Environment Variables

This project uses a `.env` file for API access:

```
HF_TOKEN=your_huggingface_token
```

### Models Used

- **LLM**: Qwen2.5-7B-Instruct  
- **VLM**: Qwen2.5-VL-7B-Instruct  

### OCR Engines

- Tesseract OCR  
- PaddleOCR  
- EasyOCR  
- docTR  
- OpenOCR  

### Remarks

- Multiple OCR engines are evaluated to analyze their impact on downstream extraction  
- The LLM processes OCR text to generate structured JSON outputs  
- The VLM directly processes document images without OCR  
## Author

Ehsan Qader

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.