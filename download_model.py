import modal
from pathlib import Path


CACHE_PATH = "/model_cache"
MODELS_DIR = "/models"
volume = modal.Volume.from_name("models", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install("easyocr")

app = modal.App(name="ffmemes-ocr-modal", image=image)


@app.function(
    image=image,
    volumes={MODELS_DIR: volume},
    gpu="any",
)
def download_model(
    mod: str = "ru",
    dir: str = MODELS_DIR,
    detect: bool = False,
    recognize: bool = False,
):
    import easyocr  # For performing optical character recognition

    # this needs to run only once to load the model into memory
    import torch

    match mod:
        case "ru":
            model = easyocr.Reader(
                ["en", "ru"],
                model_storage_directory=dir,
                gpu=True if torch.cuda.is_available() else False,
                detector=detect,
                recognizer=recognize,
                download_enabled=True,
            )
        case _:
            model = easyocr.Reader(
                ["en"],
                model_storage_directory=dir,
                gpu=True if torch.cuda.is_available() else False,
                detector=detect,
                recognizer=recognize,
                download_enabled=True,
            )
    # model_map = {
    #     "ru": "model_ru_en",
    #     "es": "model_es_en",
    #     "pt_br": "model_pt_en",
    #     "uz": "model_uz_en",
    #     "fr": "model_fr_en",
    #     "fa": "model_fa_en",
    #     "de": "model_de_en",
    #     "id": "model_id_en",
    #     "en": "model_en",
    # }

    return model


@app.local_entrypoint()
def main():
    download_model.remote()
    download_model.remote(mod="en")
