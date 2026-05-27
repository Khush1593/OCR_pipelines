# OCR Utilities (PaddleOCR + Tesseract)

This folder contains two standalone Python scripts to run OCR on **images** (PNG/JPG/etc.) and **PDFs**:

- `paddleocr_pipeline.py` — OCR using **PaddleOCR** (good accuracy; downloads models on first run).
- `teserractocr_pipeline.py` — OCR using **Tesseract** via `pytesseract` (requires Tesseract installed on your OS).

Both scripts:
- Detect text boxes
- Save an image with drawn boxes
- Save extracted text to disk
- Support PDFs by converting pages to images with `pdf2image`

## Prerequisites

### Python
- Python 3.10+ (this repo already contains a local venv at `paddle_env/`, which is Python 3.12)

### Linux system packages (required for PDFs + Tesseract)

Install these once at the OS level:

- **Poppler** (required by `pdf2image`)
- **Tesseract OCR engine** (required for `teserractocr_pipeline.py`)

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
```

If OpenCV import errors mention `libGL.so.1`, also install:

```bash
sudo apt-get install -y libgl1
```

## Installation

### Option A: Use the included virtual environment (fastest)

```bash
source paddle_env/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### Option B: Create your own venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

## How to run

Both scripts currently use a hard-coded `input_file` at the bottom of the file.

1) Open the script
2) Change `input_file = "..."` to your file path
3) Run the script

### 1) PaddleOCR pipeline

Edit `input_file` in `paddleocr_pipeline.py`, then run:

```bash
source paddle_env/bin/activate
python paddleocr_pipeline.py
```

**Outputs** (written to the current working directory):

- For images:
  - `output_image.jpg` (image with boxes)
  - `output_image.txt` (plain extracted text)
  - `output_image.md` (structured markdown)

- For PDFs (one set per page):
  - `output_page_1.jpg`, `output_page_1.txt`, `output_page_1.md`, ...

Notes:
- First run may download PaddleOCR models and take longer.
- The script disables MKL-DNN via `FLAGS_use_mkldnn=0` to avoid common CPU crashes.

### 2) Tesseract pipeline

Edit `input_file` in `teserractocr_pipeline.py`, then run:

```bash
source paddle_env/bin/activate
python teserractocr_pipeline.py
```

**Outputs** (written to the current working directory):

- For images:
  - `output_image.jpg`
  - `output_image.txt`

- For PDFs (one set per page):
  - `output_page_1.jpg`, `output_page_1.txt`, ...

#### If Tesseract is not detected

If you see `TesseractNotFoundError` (or similar), set the path explicitly near the top of `teserractocr_pipeline.py`:

```python
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
```

## Troubleshooting

- **`pdf2image` errors** like “Unable to get page count”:
  - Install Poppler (`sudo apt-get install poppler-utils`).

- **Tesseract not found**:
  - Install it (`sudo apt-get install tesseract-ocr`) or set `pytesseract.pytesseract.tesseract_cmd`.

- **Model download / slow first run (PaddleOCR)**:
  - First execution downloads weights; re-runs are faster.
