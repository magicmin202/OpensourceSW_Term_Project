"""
[팀원 B 담당] 확산 과정 (노이즈 스케줄 + 학습 loss)

Forward process: 깨끗한 이미지 x0 에 timestep t 만큼 노이즈를 섞어 x_t 를 만든다.
학습 목표: 모델이 x_t 를 보고 "섞인 노이즈"를 맞히도록 MSE loss 를 줄인다. (DDPM, Ho et al. 2020)

Classifier-Free Guidance 학습을 위해, 일정 확률(p_uncond)로 라벨을 NULL_CLASS 로 바꿔
"조건 없이 생성"하는 법도 같은 모델이 함께 배우게 한다. (Ho & Salimans 2022)

다른 팀원과의 약속(인터페이스):
    scheduler = build_scheduler("ddpm")       # 또는 "ddim"
    loss = diffusion_loss(model, scheduler, x0, y)
"""
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from .model import NULL_CLASS

SCHEDULERS = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler}


def build_scheduler(kind: str = "ddpm",
                    num_train_timesteps: int = 1000,
                    beta_schedule: str = "linear"):
    """
    kind: "ddpm" (느리지만 정석) / "ddim" (적은 step 으로 빠르게 샘플링)
    beta_schedule: "linear" / "squaredcos_cap_v2"(cosine) 등 diffusers 지원 값
    """
    if kind not in SCHEDULERS:
        raise ValueError(f"kind must be one of {list(SCHEDULERS)}, got {kind!r}")
    return SCHEDULERS[kind](num_train_timesteps=num_train_timesteps,
                            beta_schedule=beta_schedule)
