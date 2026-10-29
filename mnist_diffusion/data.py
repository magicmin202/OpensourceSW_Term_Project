"""
[팀원 C 담당] 데이터 로더

torchvision 의 MNIST 를 읽어 diffusion 학습에 맞게 전처리한다.
- 28x28 -> 32x32 (U-Net 이 해상도를 절반씩 줄이므로 2의 거듭제곱이 편함)
- 픽셀 값 [0, 1] -> [-1, 1] (가우시안 노이즈와 스케일을 맞춤)

데이터 위치: <root>/MNIST/raw/  (레포의 data/ 폴더, git 에는 올리지 않음)
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transform(image_size: int = 32):
    pad = (image_size - 28) // 2
    return transforms.Compose([
        transforms.Pad(pad),
        transforms.ToTensor(),                 # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
    ])


def get_dataloader(root: str = "data",
                   batch_size: int = 128,
                   train: bool = True,
                   image_size: int = 32,
                   num_workers: int = 0) -> DataLoader:
    """
    MNIST DataLoader. 파일이 없으면 자동 다운로드(download=True)를 시도한다.
    Windows 에서는 num_workers=0 이 가장 안전하다.
    """
    dataset = datasets.MNIST(root=root, train=train, download=True,
                             transform=get_transform(image_size))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, drop_last=train,
                      pin_memory=False)
