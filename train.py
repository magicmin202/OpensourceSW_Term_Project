"""
[팀원 C 담당] 학습 스크립트

실행 예)
    python train.py                                  # configs/default.yaml 그대로
    python train.py --epochs 3 --batch-size 64       # 일부만 바꿔서
    python train.py --max-steps 20                   # 코드가 도는지만 빠르게 확인

결과물
    outputs/checkpoints/last/   : 마지막 모델 (sample.py 가 읽음)
    outputs/samples/epoch_XX.png: epoch 마다 0~9 숫자를 생성해 본 결과
    outputs/loss.csv            : step 별 loss 기록
"""
import argparse
import os
import shutil

import torch
from tqdm import tqdm

from mnist_diffusion.config import get_device, load_config
from mnist_diffusion.data import get_dataloader
from mnist_diffusion.model import build_unet, count_parameters, save_model
from mnist_diffusion.pipeline import generate
from mnist_diffusion.scheduler import build_scheduler, diffusion_loss
from mnist_diffusion.visualize import save_grid


def parse_args():
    p = argparse.ArgumentParser(description="Train class-conditional diffusion on MNIST")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--device")
    p.add_argument("--output-dir")
    p.add_argument("--max-steps", type=int, default=None,
                   help="이 step 수만큼만 학습하고 종료 (동작 확인용)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tc = cfg["train"]
    for key, attr in [("epochs", "epochs"), ("batch_size", "batch_size"), ("lr", "lr"),
                      ("device", "device"), ("output_dir", "output_dir")]:
        if getattr(args, attr) is not None:
            tc[key] = getattr(args, attr)

    torch.manual_seed(tc["seed"])
    device = get_device(tc["device"])
    out_dir = tc["output_dir"]
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    print(f"device: {device}")

    # --- 데이터 (팀원 C) ---
    dc = cfg["data"]
    loader = get_dataloader(dc["root"], tc["batch_size"], train=True,
                            image_size=dc["image_size"], num_workers=dc["num_workers"])

    # --- 모델 (팀원 A) ---
    mc = cfg["model"]
    model = build_unet(dc["image_size"], mc["block_out_channels"],
                       mc["layers_per_block"]).to(device)
    print(f"U-Net parameters: {count_parameters(model) / 1e6:.2f}M")

    # --- 확산 과정 (팀원 B) ---
    df = cfg["diffusion"]
    train_sched = build_scheduler("ddpm", df["num_train_timesteps"], df["beta_schedule"])
    sc = cfg["sample"]
    sample_sched = build_scheduler(sc["sampler"], df["num_train_timesteps"], df["beta_schedule"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"])
    preview_labels = [d for _ in range(4) for d in range(10)]   # 4행 x 10열, 열마다 숫자 0~9

    loss_log = open(os.path.join(out_dir, "loss.csv"), "w", encoding="utf-8")
    loss_log.write("step,epoch,loss\n")

    step = 0
    done = False
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{tc['epochs']}")
        for x0, y in pbar:
            x0, y = x0.to(device), y.to(device)
            loss = diffusion_loss(model, train_sched, x0, y, p_uncond=df["p_uncond"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            loss_log.write(f"{step},{epoch},{loss.item():.6f}\n")
            if args.max_steps and step >= args.max_steps:
                done = True
                break

        # epoch 끝: 체크포인트 + 미리보기 이미지
        ckpt = os.path.join(out_dir, "checkpoints", "last")
        save_model(model, ckpt)
        shutil.copy(args.config, os.path.join(ckpt, "train_config.yaml"))

        if epoch % tc["sample_every"] == 0 or done:
            imgs = generate(model, sample_sched, preview_labels,
                            guidance_scale=sc["guidance_scale"],
                            num_inference_steps=sc["num_inference_steps"],
                            image_size=dc["image_size"], seed=0)
            path = os.path.join(out_dir, "samples", f"epoch_{epoch:02d}.png")
            save_grid(imgs, path, nrow=10)
            print(f"saved {path}")
        if done:
            break

    loss_log.close()
    print(f"done. checkpoint -> {os.path.join(out_dir, 'checkpoints', 'last')}")


if __name__ == "__main__":
    main()
