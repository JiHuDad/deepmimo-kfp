# pytorch-cpu: python-cpu + PyTorch (CPU-only)
ARG IMAGE_TAG=latest
FROM localhost:5000/python-cpu:${IMAGE_TAG}

COPY offline-packages/wheels/ /tmp/wheels/
COPY mlops_platform/base-images/pytorch-cpu/requirements.txt /tmp/requirements.txt

RUN pip install --no-index --find-links=/tmp/wheels/ \
        -r /tmp/requirements.txt \
    && rm -rf /tmp/wheels/ /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1
