import modal
from pathlib import Path

import modal.gpu

# Define the image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .pip_install("torch")
    .pip_install("pillow")
    .pip_install("easyocr")
    .pip_install("fastapi[standard]")
    .run_commands("echo 'ready to go!'")
    # .run_function(download_model("ru"), gpu="any")
    # .run_function(download_model("en"), gpu="any")
)

# Create a new app
app = modal.App(name="ffmemes-ocr-modal", image=image)
# Define the cache path and models directory
CACHE_PATH = "/model_cache"
MODELS_DIR = "/models"

# Defining the storage volume
try:
    volume = modal.Volume.lookup("models", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_model.py")

# @app.function(
#     image=image,
#     volumes={MODELS_DIR: volume},
#     gpu="any",
# )
# def download_model(self, mod: str = "ru", dir: str = MODELS_DIR):
#     import easyocr
#     import torch

#     match mod:
#         case "ru":
#             model = easyocr.Reader(
#                 ["en", "ru"],
#                 model_storage_directory=dir,
#                 gpu=True if torch.cuda.is_available() else False,
                
#             )
#         case _:
#             model = easyocr.Reader(
#                 ["en"],
#                 model_storage_directory=dir,
#                 gpu=True if torch.cuda.is_available() else False,
#             )
#     # model_map = {
#     #     "ru": "model_ru_en",
#     #     "es": "model_es_en",
#     #     "pt_br": "model_pt_en",
#     #     "uz": "model_uz_en",
#     #     "fr": "model_fr_en",
#     #     "fa": "model_fa_en",
#     #     "de": "model_de_en",
#     #     "id": "model_id_en",
#     #     "en": "model_en",
#     # }

#     return model

LANG_MAP = {
    "Russian": "ru",
    "English": "en",
    "Spanish": "es",
    "Portuguese": "pt",
    "Uzbek": "uz",
    "French": "fr",
    "Farsi": "fa",
    "German": "de",
    "Indonesian": "id",
}

LANG_MAP_REV = {v: k for k, v in LANG_MAP.items()}

# App class
@app.cls(image=image, gpu=modal.gpu.L4(count=1), volumes={MODELS_DIR: volume})
class WebApp:
    # Entry point: loading the models
    @modal.enter()
    def startup(self):
        import time
        start = time.time()
        from pathlib import Path
        import easyocr
        import torch

        print(f"Startup() loading libs: {time.time() - start:.3f} seconds")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_map = {
            "ru": "model_ru_en",
            "es": "model_es_en",
            "pt_br": "model_pt_en",
            "uz": "model_uz_en",
            "fr": "model_fr_en",
            "fa": "model_fa_en",
            "de": "model_de_en",
            "id": "model_id_en",
            "en": "model_en",
        }

        self.model_ru = easyocr.Reader(
            ["en", "ru"],
            model_storage_directory=MODELS_DIR,
            gpu=True if torch.cuda.is_available() else False,
            download_enabled=False
        )
        print(f"Startup() loading model_ru: {time.time() - start:.3f} seconds")
        self.model_en = easyocr.Reader(
            ["en"],
            model_storage_directory=MODELS_DIR,
            gpu=True if torch.cuda.is_available() else False,
            download_enabled=False
        )
        print(f"Startup() loading model_en: {time.time() - start:.3f} seconds")

        print("🏁 Starting!")

    from fastapi import UploadFile, File, Body
    from typing import List, Union, Tuple
    
    # More precise typing for OCR results
    BBox = List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    OCRResult = List[Tuple[BBox, str]]  # List of [bbox, text] pairs

    # Prediction REST endpoint which we'll use to perform OCR
    @modal.web_endpoint(method="POST", docs=True)
    # async def predict(self, image: UploadFile = File(...), lang: str = "en") -> str:
    async def predict(self, image: bytes = Body(...), lang: str = "ru") -> Union[str, OCRResult]:
        """
        Performs optical character recognition (OCR) on the provided image file.

        Parameters:
        image (File): This should be a .png, .jpg, or other image file; or a URL when calling by API.
        lang (str): The language to use for OCR. Can be either 'ru'/Russian or 'en'/English. Defaults to English.

        Returns:
        str: The OCR output as a string. If the language is Russian, the function uses a model trained on both English and Russian. If the language is English or any other value, the function uses a model trained only on English.

        Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the language is not supported.
        """
        import time
        start = time.time()
        from PIL import Image, UnidentifiedImageError  # For working with images
        import io
        import torch
        from numpy import asarray
        import base64

        print(f'Time to load libs: {time.time() - start:.3f} seconds')
        # try:
        #     image_bytes = await image.read()
        # except Exception as e:
        #     return f"Error reading image: {str(e)}"

        # Open the file in binary mode and read it into a BytesIO object
        
        try:
            img = Image.open(io.BytesIO(image))
        except UnidentifiedImageError as e:
            print(f"UnidentifiedImageError: {e}")
            return {"error": "Cannot identify image file"}
        
        img = img.convert("L")

        img = asarray(img)
        # img = asarray(Image.open(io.BytesIO(image)).convert("L"))
        print(f"Time taken to read image: {time.time() - start:.3f} seconds")
        out = ""

        match lang:
            case "ru" | "Russian":
                out = self.model_ru.readtext(img, paragraph=True)
            # case "en" | "English":
            #     out = model_en.readtext(img, paragraph=True)
            # case "es" | "Spanish":
            #     out = model_es_en.readtext(img, paragraph=True)
            # case "pt" | "pt_br" | "Portuguese / Brazilian":
            #     out = model_pt_en.readtext(img, paragraph=True)
            # case "uz" | "Uzbek":
            #     out = model_uz_en.readtext(img, paragraph=True)
            # case "fr" | "French":
            #     out = model_fr_en.readtext(img, paragraph=True)
            # case "fa" | "Farsi":
            #     out = model_fa_en.readtext(img, paragraph=True)
            # case "de" | "German":
            #     out = model_de_en.readtext(img, paragraph=True)
            # case "id" | "Indonesian":
            #     out = model_id_en.readtext(img, paragraph=True)
            case _:
                out = self.model_en.readtext(img, paragraph=True)

        print(f"Time taken to OCR: {time.time() - start:.3f} seconds")
        print(f"Output: {out}")
        return out