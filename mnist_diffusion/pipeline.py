"""
[팀원 B 담당] 샘플링 (Reverse process + Classifier-Free Guidance)

순수 노이즈에서 시작해 모델이 예측한 노이즈를 조금씩 빼면서 이미지를 만든다.
CFG: 같은 x_t 에 대해 "조건 있음"과 "조건 없음" 예측을 모두 구한 뒤
     eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
     로 조건 쪽을 더 강하게 따르도록 한다. guidance_scale=1 이면 일반 조건부 생성.

다른 팀원과의 약속(인터페이스):
    images = generate(model, scheduler, labels, guidance_scale=3.0, num_inference_steps=50)
    # images: [N, 1, H, W], 값 범위 [0, 1]
"""
from typing import List, Optional, Tuple, Union

import torch

from .model import NULL_CLASS


@torch.no_grad()
def generate(model,
             scheduler,
             labels: Union[torch.Tensor, List[int]],
             guidance_scale: float = 3.0,
             num_inference_steps: int = 50,
             image_size: int = 32,
             seed: Optional[int] = None,
             return_trajectory: bool = False,
             trajectory_every: int = 5,
             ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    labels: 만들고 싶은 숫자들. 예) [7, 7, 3] -> 7 두 장, 3 한 장
    return_trajectory=True 이면 중간 과정 이미지 리스트도 함께 반환 (GIF 용)
    """
    device = next(model.parameters()).device
    model.eval()

    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    n = y.shape[0]
    y_null = torch.full_like(y, NULL_CLASS)

    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    x = torch.randn((n, 1, image_size, image_size), generator=g).to(device)

    scheduler.set_timesteps(num_inference_steps)
    trajectory = []
    for i, t in enumerate(scheduler.timesteps):
        t_batch = torch.full((2 * n,), int(t), device=device, dtype=torch.long)
        eps = model(torch.cat([x, x]), t_batch,
                    class_labels=torch.cat([y, y_null])).sample
        eps_cond, eps_uncond = eps.chunk(2)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        x = scheduler.step(eps, t, x).prev_sample

        if return_trajectory and (i % trajectory_every == 0):
            trajectory.append(to_unit_range(x).cpu())

    images = to_unit_range(x).cpu()
    if return_trajectory:
        trajectory.append(images)
        return images, trajectory
    return images


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]"""
    return (x.clamp(-1, 1) + 1) / 2
