# deepmimo-trainer: base + PyTorch (CPU)
FROM localhost:5000/deepmimo-base:latest

COPY offline-packages/wheels/ /tmp/wheels/
COPY docker/trainer/requirements.txt /tmp/requirements.txt

RUN pip install --no-index --find-links=/tmp/wheels/ \
        -r /tmp/requirements.txt \
    && rm -rf /tmp/wheels/ /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1
