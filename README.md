# MNIST Diffusion: 원하는 숫자를 그려주는 Diffusion 모델

> Gachon University 오픈소스SW 텀프로젝트
> Hugging Face **diffusers** 기반 클래스 조건부 DDPM + Classifier-Free Guidance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)

## 1. 프로젝트 소개

**MNIST Diffusion**은 "숫자 7 그려줘"처럼 원하는 숫자를 지정하면, 순수한 노이즈에서 시작해 단계적으로 노이즈를 걷어내며 해당 숫자의 손글씨 이미지를 생성하는 오픈소스 프로젝트입니다.

이미지 생성 AI(Stable Diffusion 등)의 핵심 원리인 **Diffusion 모델**을 직접 학습시켜 볼 수 있도록 만들었습니다. 이미지 크기가 작고(32x32) 모델이 가벼워서(약 3.9M 파라미터) Colab 무료 GPU로도 20분 안팎이면 학습할 수 있습니다.

### 동작 원리

```
학습:  깨끗한 숫자 이미지 ──(노이즈 추가)──▶ 노이즈 섞인 이미지 ──▶ U-Net이 "섞인 노이즈"를 예측하도록 학습
생성:  순수 노이즈 ──▶ U-Net이 예측한 노이즈를 조금씩 제거 (50~1000 step) ──▶ 숫자 이미지
```

## 2. 주요 기능

| 기능 | 설명 |
|---|---|
| **조건부 생성** | 0~9 중 원하는 숫자를 지정해서 생성 (`--digits 2026`) |
| **Classifier-Free Guidance** | guidance scale로 "조건을 얼마나 강하게 따를지" 조절 (`--guidance 5`) |
| **두 가지 샘플러** | 정석 DDPM(1000 step) / 빠른 DDIM(50 step) 선택 (`--sampler ddim`) |
| **생성 과정 시각화** | 노이즈가 숫자로 변해가는 과정을 GIF로 저장 (`--gif`) |
| **학습 모니터링** | epoch마다 미리보기 이미지와 loss 기록(`loss.csv`) 자동 저장 |
| **재사용 가능한 패키지** | `mnist_diffusion` 패키지로 모듈화되어 다른 코드에서 import 가능 |
| **설정 파일 기반** | 모든 하이퍼파라미터를 `configs/default.yaml` 한 곳에서 관리 |

## 3. 데모

> 학습 후 결과 이미지를 `assets/` 폴더에 넣으면 아래에 표시됩니다.

| 0~9 생성 결과 | 생성 과정 (노이즈 → 숫자) |
|:---:|:---:|
| ![samples](assets/samples.png) | ![process](assets/process.gif) |

## 4. 설치 가이드 (Environment & Installation)

### 개발 및 테스트 환경

| 항목 | 버전 |
|---|---|
| OS | Windows 10/11, Ubuntu 22.04, Google Colab |
| Python | 3.10 이상 |
| GPU | 권장 (Colab T4 기준 10 epoch 약 10~20분), CPU로도 실행 가능 |

### 사용한 패키지

| 패키지 | 버전 | 용도 |
|---|---|---|
| torch | >= 2.1 | 딥러닝 프레임워크 |
| torchvision | >= 0.16 | MNIST 데이터셋, 이미지 격자 |
| diffusers | >= 0.30 | `UNet2DModel`, `DDPMScheduler`, `DDIMScheduler` |
| accelerate | >= 0.30 | diffusers 모델 로딩 |
| pyyaml | >= 6.0 | 설정 파일 |
| tqdm | >= 4.66 | 학습 진행바 |
| pillow | >= 10.0 | 이미지 / GIF 저장 |

개발 중 테스트한 버전: torch 2.14.1, torchvision 0.29.1, diffusers 0.42.1

### 설치

```bash
git clone https://github.com/magicmin202/OpensourceSW_Term_Project.git
cd OpensourceSW_Term_Project

# (선택) 가상환경
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

GPU를 사용하려면 [PyTorch 공식 설치 페이지](https://pytorch.org/get-started/locally/)에서 본인 CUDA 버전에 맞는 torch를 먼저 설치하세요.

### Google Colab에서 실행

```python
!git clone https://github.com/magicmin202/OpensourceSW_Term_Project.git
%cd OpensourceSW_Term_Project
!pip install -r requirements.txt
!python train.py
```
(런타임 → 런타임 유형 변경 → T4 GPU 선택)

MNIST 데이터는 처음 실행할 때 `data/` 폴더에 자동으로 다운로드됩니다.

## 5. 실행 방법 (Quick Start / Usage)

### Quick Start

```bash
python train.py --max-steps 20     # 1~2분 안에 끝나는 동작 확인
python train.py                    # 본 학습 (기본 10 epoch)
python sample.py --gif             # 0~9 생성 + 생성 과정 GIF
```

### 학습 (`train.py`)

```bash
python train.py                          # configs/default.yaml 설정 그대로
python train.py --epochs 20 --lr 1e-4    # 일부 설정만 바꾸기
python train.py --device cpu             # 장치 지정 (auto | cuda | cpu | mps)
```

| 결과물 | 위치 |
|---|---|
| 모델 체크포인트 | `outputs/checkpoints/last/` |
| epoch별 미리보기 | `outputs/samples/epoch_XX.png` |
| loss 기록 | `outputs/loss.csv` |

### 생성 (`sample.py`)

```bash
python sample.py                                  # 0~9 각 8장
python sample.py --digits 2026 --n 4              # 원하는 숫자 조합
python sample.py --digits 7 --n 16 --guidance 5   # guidance를 세게
python sample.py --guidance 1                     # guidance 없이 (비교용)
python sample.py --sampler ddpm --steps 1000      # 정석 DDPM 샘플링 (느림)
python sample.py --gif --out outputs/process.png  # 생성 과정 GIF도 저장
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--digits` | 생성할 숫자들 | `0123456789` |
| `--n` | 숫자 하나당 장수 | 8 |
| `--guidance` | CFG guidance scale (1 = 사용 안 함) | 3.0 |
| `--sampler` | `ddpm` 또는 `ddim` | ddim |
| `--steps` | 샘플링 step 수 | 50 |
| `--gif` | 생성 과정 GIF 저장 | 꺼짐 |
| `--ckpt` | 체크포인트 경로 | `outputs/checkpoints/last` |

### 라이브러리로 사용하기

```python
from mnist_diffusion.model import load_model
from mnist_diffusion.scheduler import build_scheduler
from mnist_diffusion.pipeline import generate
from mnist_diffusion.visualize import save_grid

model = load_model("outputs/checkpoints/last", device="cuda")
scheduler = build_scheduler("ddim")
images = generate(model, scheduler, labels=[1, 2, 3], guidance_scale=3.0)
save_grid(images, "my_digits.png", nrow=3)
```

## 6. 프로젝트 구조

```
.
├── mnist_diffusion/          # 직접 만든 Python 패키지
│   ├── model.py              # U-Net 노이즈 예측 모델
│   ├── scheduler.py          # 노이즈 스케줄, 학습 loss
│   ├── pipeline.py           # 샘플링 + Classifier-Free Guidance
│   ├── data.py               # MNIST 데이터 로더
│   ├── visualize.py          # 이미지 격자 / GIF 저장
│   └── config.py             # 설정 파일 읽기
├── configs/default.yaml      # 하이퍼파라미터
├── train.py                  # 학습 스크립트
├── sample.py                 # 생성 스크립트
├── assets/                   # README 데모 이미지
├── requirements.txt
└── LICENSE
```

## 7. 실험 결과

> 실험 후 채워 넣을 예정입니다.

- 모델 크기별 비교 (파라미터 수 / 학습 시간 / 생성 품질)
- DDPM vs DDIM, step 수별 생성 속도와 품질
- guidance scale별 생성 결과 비교
- 학습 loss 곡선, epoch별 생성 결과 변화

## 8. 팀원 역할 분담 및 기여도

| 이름 | 학번 | GitHub | 역할 | 담당 파일 |
|---|---|---|---|---|
| 이승민 (팀장) | 202337623 | [@magicmin202](https://github.com/magicmin202) | 레포 관리, PR 리뷰 및 merge, 프로젝트 설정 / **데이터·학습·생성·시각화** | `README.md`, `LICENSE`, `configs/`, `mnist_diffusion/config.py`, `mnist_diffusion/data.py`, `mnist_diffusion/visualize.py`, `train.py`, `sample.py` |
| (팀원 A 이름) | (학번) | (@GitHubID) | **노이즈 예측 모델 (U-Net)** | `mnist_diffusion/model.py` |
| (팀원 B 이름) | (학번) | (@GitHubID) | **확산 과정 · 샘플링 · Classifier-Free Guidance** | `mnist_diffusion/scheduler.py`, `mnist_diffusion/pipeline.py` |

### 세부 기여 내용

- **이승민 (팀장)**
  - GitHub 레포지토리 생성, 협업 규칙(브랜치 → Pull Request → 리뷰 후 merge) 운영
  - 프로젝트 공통 설정(README, LICENSE, requirements, 설정 파일, 패키지 구조) 작성
  - MNIST 데이터 로더, 학습 루프, 생성 CLI, 결과 시각화(이미지 격자, GIF) 구현
  - 전체 모듈 통합 및 학습 실행, 데모 결과 정리
- **팀원 A**
  - diffusers `UNet2DModel` 기반 클래스 조건부 U-Net 설계
  - CFG용 "조건 없음" 라벨 임베딩, 모델 저장/불러오기 기능 구현
- **팀원 B**
  - DDPM / DDIM 노이즈 스케줄러 구성, 노이즈 예측 학습 loss 구현
  - Classifier-Free Guidance 학습(라벨 드롭아웃)과 샘플링 파이프라인 구현

### 협업 방식

1. 팀장이 공통 설정을 `main`에 올림
2. 각 팀원이 `feature/<담당>` 브랜치에서 작업 후 Pull Request
3. 팀장이 리뷰 후 `main`에 merge
4. commit 내역은 [Commits](https://github.com/magicmin202/OpensourceSW_Term_Project/commits/main) 및 [Pull requests](https://github.com/magicmin202/OpensourceSW_Term_Project/pulls?q=is%3Apr)에서 확인할 수 있습니다.

## 9. 라이선스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다. 자유롭게 사용, 수정, 배포할 수 있습니다.

사용한 외부 라이브러리와 데이터의 라이선스:
- PyTorch / torchvision: BSD-3-Clause
- Hugging Face diffusers / accelerate: Apache 2.0
- MNIST 데이터셋: Yann LeCun, Corinna Cortes, Christopher J.C. Burges (CC BY-SA 3.0)

## 10. 참고 자료

- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. https://arxiv.org/abs/2006.11239
- Song, Meng, Ermon. *Denoising Diffusion Implicit Models*. ICLR 2021. https://arxiv.org/abs/2010.02502
- Ho, Salimans. *Classifier-Free Diffusion Guidance*. 2022. https://arxiv.org/abs/2207.12598
- Hugging Face diffusers 문서: https://huggingface.co/docs/diffusers
- Hugging Face Diffusion Models Course: https://github.com/huggingface/diffusion-models-class
- MNIST 데이터셋 (torchvision): https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
