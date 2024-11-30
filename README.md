# Modal EasyOCR

A FastAPI web service that performs Optical Character Recognition (OCR) using EasyOCR, deployed on Modal.com's infrastructure.

## Features

- Supports English and Russian text recognition
- GPU-accelerated inference
- REST API endpoint for easy integration
- Persistent model storage using Modal volumes
- Automatic model downloading and caching

## Prerequisites

- Python 3.11 or higher
- Modal CLI and account
- Required Python packages: requests, numpy, torch, pillow, easyocr, fastapi

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/modal-easyocr.git
cd modal-easyocr
```

2. Install Modal CLI:
```bash
pip install modal
```

3. Login to Modal:

```bash
modal token new
```

4. Download the models:

```bash
modal run download_model.py
```

5. Deploy the service:

```bash
modal deploy final.py
```

## Usage
Send POST requests to the API endpoint with an image file:
```python
import requests

url = "https://yourusername--ffmemes-ocr-modal-predict.modal.run"
files = {"image": open("image.png", "rb")}
params = {"lang": "en"}  # or "ru" for Russian

response = requests.post(url, files=files, params=params)
print(response.json())
```