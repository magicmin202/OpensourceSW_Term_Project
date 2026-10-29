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


def build_unet(image_size: int = 32,
               block_out_channels=(32, 64, 128),
               layers_per_block: int = 2,
               num_classes: int = NUM_CLASSES) -> UNet2DModel:
    """MNIST(32x32, 흑백)에 맞춘 작은 클래스 조건부 U-Net 을 만든다."""
    block_out_channels = tuple(block_out_channels)
    n = len(block_out_channels)
    # 해상도가 가장 낮은 단계에만 self-attention 을 넣어 계산량을 줄인다.
    down_types = ("DownBlock2D",) * (n - 1) + ("AttnDownBlock2D",)
    up_types = ("AttnUpBlock2D",) + ("UpBlock2D",) * (n - 1)

    return UNet2DModel(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_types,
        up_block_types=up_types,
        num_class_embeds=num_classes + 1,   # +1 = NULL_CLASS
    )


def count_parameters(model: torch.nn.Module) -> int:
    """학습 가능한 파라미터 수."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: UNet2DModel, path: str) -> None:
    """diffusers 형식(config.json + 가중치)으로 저장."""
    model.save_pretrained(path)


def load_model(path: str, device: str = "cpu") -> UNet2DModel:
    """save_model 로 저장한 모델을 불러온다."""
    return UNet2DModel.from_pretrained(path).to(device).eval()


if __name__ == "__main__":
    # 단독 테스트: python -m mnist_diffusion.model
    model = build_unet()
    x = torch.randn(4, 1, 32, 32)
    t = torch.randint(0, 1000, (4,))
    y = torch.tensor([0, 3, 7, NULL_CLASS])
    out = model(x, t, class_labels=y).sample
    print("output shape:", tuple(out.shape))
    print(f"parameters: {count_parameters(model) / 1e6:.2f}M")
