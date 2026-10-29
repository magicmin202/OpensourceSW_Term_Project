"""
[팀원 C 담당] 결과 시각화

- save_grid: 생성 이미지를 격자 한 장(PNG)으로 저장
- save_gif : 노이즈 -> 숫자로 변해가는 과정을 GIF 로 저장 (README 데모용)
"""
import os
from typing import List

import torch
from PIL import Image
from torchvision.utils import make_grid


def _to_pil(grid: torch.Tensor, scale: int = 1) -> Image.Image:
    arr = (grid.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(arr)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def save_grid(images: torch.Tensor, path: str, nrow: int = 10, scale: int = 2) -> None:
    """images: [N, 1, H, W] (값 범위 [0, 1])"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
    _to_pil(grid, scale).save(path)


def save_gif(trajectory: List[torch.Tensor], path: str, nrow: int = 10,
             scale: int = 2, duration_ms: int = 120, hold_last_ms: int = 1500) -> None:
    """trajectory: generate(..., return_trajectory=True) 가 돌려준 이미지 리스트"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frames = [_to_pil(make_grid(x, nrow=nrow, padding=2, pad_value=1.0), scale).convert("P")
              for x in trajectory]
    durations = [duration_ms] * (len(frames) - 1) + [hold_last_ms]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
