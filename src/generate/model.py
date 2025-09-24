from __future__ import annotations

from typing import Tuple

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


def _pick_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(precision: str | None = None, device: str = "cpu") -> torch.dtype:
    # On Apple MPS, use float32 to avoid NaNs/black outputs.
    if device == "mps":
        return torch.float32
    if precision == "fp32" or device == "cpu":
        return torch.float32
    return torch.float16


def _set_scheduler(pipeline: DiffusionPipeline, name: str) -> None:
    if name.lower() in {"dpmpp_2m", "dpm-solver++-2m", "dpm-solver++"}:
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras=True)
        return
    # default: keep as-is


def load_pipeline(
    model_id: str, device: str | None, precision: str | None, scheduler: str
) -> Tuple[DiffusionPipeline, str]:
    the_device = _pick_device(device)
    dtype = _pick_dtype(precision, the_device)
    pipe = DiffusionPipeline.from_pretrained(model_id, dtype=dtype, safety_checker=None)
    _set_scheduler(pipe, scheduler)
    if the_device in {"cuda", "mps"}:
        pipe = pipe.to(the_device)
        pipe.enable_attention_slicing()
    return pipe, the_device
