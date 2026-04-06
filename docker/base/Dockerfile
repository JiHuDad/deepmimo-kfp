# deepmimo-base: DeepMIMO + 공통 과학 계산 환경
# 폐쇄망: 오프라인 wheels 사용, FROM은 로컬 레지스트리 참조
FROM 192.168.1.112:5000/python:3.12-slim

WORKDIR /app

# 빌드 컨텍스트에서 복사한 오프라인 wheels
COPY offline-packages/wheels/ /tmp/wheels/
COPY docker/base/requirements.txt /tmp/requirements.txt

RUN pip install --no-index --find-links=/tmp/wheels/ \
        -r /tmp/requirements.txt \
    && rm -rf /tmp/wheels/ /tmp/requirements.txt \
    && mkdir -p /data/scenarios /data/output

ENV PYTHONUNBUFFERED=1
