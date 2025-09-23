import torch
from PIL import Image
from torchvision import transforms


def get_inception_preprocess(image_size: int = 299) -> transforms.Compose:
    """
    Preprocessing for InceptionV3-compatible inputs (FID/IS):
    - Convert to RGB
    - Resize shortest side so that a center crop yields image_size x image_size
    - CenterCrop to exact image_size
    - Convert to tensor in [0,1]
    - Normalize with Inception-style mean/std (same as ImageNet)
    """
    normalize_mean = [0.5, 0.5, 0.5]
    normalize_std = [0.5, 0.5, 0.5]

    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if isinstance(img, Image.Image) else img),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )


def preprocess_pil_image(pil_image: Image.Image, image_size: int = 299) -> torch.Tensor:
    transform = get_inception_preprocess(image_size=image_size)
    return transform(pil_image)


def preprocess_batch(pil_images: list[Image.Image], image_size: int = 299) -> torch.Tensor:
    transform = get_inception_preprocess(image_size=image_size)
    tensors = [transform(img) for img in pil_images]
    return torch.stack(tensors, dim=0)
