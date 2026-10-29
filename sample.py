"""
[팀원 C 담당] 학습된 모델로 숫자 이미지 생성

실행 예)
    python sample.py                                  # 0~9 를 각각 8장씩
    python sample.py --digits 2026 --n 4              # "2", "0", "2", "6" 각 4장
    python sample.py --digits 7 --n 16 --guidance 5   # guidance 를 세게
    python sample.py --sampler ddpm --steps 1000      # 정석 DDPM (느림)
    python sample.py --gif                            # 생성 과정 GIF 도 저장
"""
import argparse
import os

import torch

from mnist_diffusion.config import get_device, load_config
from mnist_diffusion.model import load_model
from mnist_diffusion.pipeline import generate
from mnist_diffusion.scheduler import build_scheduler
from mnist_diffusion.visualize import save_gif, save_grid


def parse_args():
    p = argparse.ArgumentParser(description="Generate MNIST digits with a trained diffusion model")
    p.add_argument("--ckpt", default="outputs/checkpoints/last")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--digits", default="0123456789", help="생성할 숫자들 (예: 2026)")
    p.add_argument("--n", type=int, default=8, help="숫자 하나당 장수")
    p.add_argument("--sampler", choices=["ddpm", "ddim"])
    p.add_argument("--steps", type=int, help="샘플링 step 수")
    p.add_argument("--guidance", type=float, help="CFG guidance scale (1 = 사용 안 함)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="outputs/generated.png")
    p.add_argument("--gif", action="store_true", help="생성 과정을 GIF 로도 저장")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    sc, df = cfg["sample"], cfg["diffusion"]
    sampler = args.sampler or sc["sampler"]
    steps = args.steps or sc["num_inference_steps"]
    guidance = sc["guidance_scale"] if args.guidance is None else args.guidance

    device = get_device(args.device)
    model = load_model(args.ckpt, device)
    scheduler = build_scheduler(sampler, df["num_train_timesteps"], df["beta_schedule"])

    digits = [int(c) for c in args.digits if c.isdigit()]
    # 열 = 숫자, 행 = 같은 숫자의 여러 샘플
    labels = [d for _ in range(args.n) for d in digits]

    result = generate(model, scheduler, labels, guidance_scale=guidance,
                      num_inference_steps=steps, image_size=model.config.sample_size,
                      seed=args.seed, return_trajectory=args.gif)
    if args.gif:
        images, trajectory = result
        gif_path = os.path.splitext(args.out)[0] + ".gif"
        save_gif(trajectory, gif_path, nrow=len(digits))
        print(f"saved {gif_path}")
    else:
        images = result

    save_grid(images, args.out, nrow=len(digits))
    print(f"saved {args.out}  ({sampler}, {steps} steps, guidance={guidance})")


if __name__ == "__main__":
    main()
