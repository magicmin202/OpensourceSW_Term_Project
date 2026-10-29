"""
mnist_diffusion: 클래스 조건부 Diffusion 으로 MNIST 손글씨 숫자 생성

모듈 구성 (담당)
- model.py      : U-Net 노이즈 예측 모델            [팀원 A]
- scheduler.py  : 노이즈 스케줄, 학습 loss           [팀원 B]
- pipeline.py   : 샘플링 + Classifier-Free Guidance [팀원 B]
- data.py       : MNIST 데이터 로더                 [팀원 C]
- visualize.py  : 이미지 격자 / GIF 저장             [팀원 C]
- config.py     : 설정 파일 읽기                     [공통]
"""
__version__ = "0.1.0"
