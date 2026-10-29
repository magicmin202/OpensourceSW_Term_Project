"""
[공통] 설정 파일(configs/*.yaml) 읽기

사용 예)
    cfg = load_config("configs/default.yaml")
    cfg["train"]["epochs"]
"""
import yaml

import torch


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(name: str = "auto") -> str:
    """'auto' 이면 cuda > mps > cpu 순서로 사용 가능한 장치를 고른다."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
