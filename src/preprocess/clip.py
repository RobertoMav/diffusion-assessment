from typing import Optional

import open_clip
import torch
from PIL import Image


def get_clip_model_and_preprocess(
    model_name: str = "ViT-B-32-quickgelu",
    pretrained: str = "laion400m_e32",
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def encode_texts(texts: list[str], model, tokenizer, device: str) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(device_type=device if device != "cpu" else "cpu", enabled=(device != "cpu")):
        tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def encode_images(pil_images: list[Image.Image], model, preprocess, device: str) -> torch.Tensor:
    pixel_batches = torch.stack([preprocess(img) for img in pil_images]).to(device)
    with torch.no_grad(), torch.autocast(device_type=device if device != "cpu" else "cpu", enabled=(device != "cpu")):
        image_features = model.encode_image(pixel_batches)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features
