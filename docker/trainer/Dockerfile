# deepmimo-trainer: base + PyTorch (CPU) + 학습 프레임워크
FROM localhost:5000/deepmimo-base:latest

COPY offline-packages/wheels/ /tmp/wheels/
COPY docker/trainer/requirements.txt /tmp/requirements.txt

RUN pip install --no-index --find-links=/tmp/wheels/ \
        -r /tmp/requirements.txt \
    && rm -rf /tmp/wheels/ /tmp/requirements.txt

# 학습 프레임워크 및 모델 코드를 이미지에 포함
# 새 모델 추가 시 models/ 에 파일을 추가하고 이미지를 재빌드하세요
COPY framework/ /app/framework/
COPY models/    /app/models/

# /app 을 Python 경로에 추가하여 'from framework...' 임포트가 동작하도록 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
