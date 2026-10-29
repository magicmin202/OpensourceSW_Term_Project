"""
노이즈 예측 신경망 (U-Net)

Hugging Face diffusers 의 UNet2DModel 을 이용해 '클래스 조건부' U-Net 을 만든다.
- 입력: 노이즈가 섞인 이미지 x_t [B, 1, 32, 32], timestep t [B], 숫자 라벨 y [B]
- 출력: x_t 안에 섞여 있는 노이즈 예측값 [B, 1, 32, 32]

Classifier-Free Guidance(CFG)를 위해 라벨 임베딩을 10개(숫자 0~9) + 1개("조건 없음")로 만든다.
"조건 없음" 라벨 번호는 NULL_CLASS(=10) 이다.

다른 팀원과의 약속(인터페이스):
    model = build_unet(cfg)
    eps_pred = model(x_t, t, class_labels=y).sample
"""
import torch
from diffusers import UNet2DModel

NUM_CLASSES = 10            # MNIST 숫자 0~9
NULL_CLASS = NUM_CLASSES    # CFG 에서 "조건 없음"을 뜻하는 라벨 번호

