# deepmimo: python-cpu + DeepMIMO 패키지
# 프로젝트 전용 이미지 (preprocess 컴포넌트에서 사용)
ARG IMAGE_TAG=latest
FROM localhost:5000/python-cpu:${IMAGE_TAG}

COPY offline-packages/wheels/ /tmp/wheels/
COPY projects/deepmimo_beam_selection/docker/deepmimo/requirements.txt /tmp/requirements.txt

RUN pip install --no-index --find-links=/tmp/wheels/ \
        -r /tmp/requirements.txt \
    && rm -rf /tmp/wheels/ /tmp/requirements.txt \
    && mkdir -p /data/scenarios

# features/ 모듈을 이미지에 포함
# preprocess 컴포넌트와 서빙 Transformer 가 동일 코드를 import 한다
COPY projects/__init__.py                                 /app/projects/__init__.py
COPY projects/deepmimo_beam_selection/__init__.py         /app/projects/deepmimo_beam_selection/__init__.py
COPY projects/deepmimo_beam_selection/features/           /app/projects/deepmimo_beam_selection/features/

# /app 을 Python 경로에 추가
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
