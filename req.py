import requests
import time

# Updated req.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

url = "https://ffmemes-adm--ffmemes-ocr-modal-webapp-predict-dev.modal.run"

# Query parameters
params = {"lang": "ru"}

# Headers for binary data
headers = {"accept": "application/json", "Content-Type": "application/octet-stream"}

# Read image as binary
with open("receipt.png", "rb") as image_file:
    image_bytes = image_file.read()

# Start measuring time
start = time.time()

# Make POST request with raw bytes, sending image directly
response = requests.post(
    url, params=params, headers=headers, data=image_bytes
)  # Send raw bytes directly

# Print response
print(response.json())
print(f"Time taken: {time.time() - start:.3f} seconds")
