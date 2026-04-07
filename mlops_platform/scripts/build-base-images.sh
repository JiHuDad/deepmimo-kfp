#!/usr/bin/env bash
# ============================================================
# build-base-images.sh
#
# 플랫폼 베이스 이미지(python-cpu, pytorch-cpu)를 빌드하고
# 로컬 레지스트리(localhost:5000)에 push한다.
#
# 전제조건:
#   - offline-packages/wheels/ 에 whl 파일이 있어야 함
#   - offline-packages/python-3.12-slim.tar 이 있어야 함
#   - docker daemon 실행 중
# ============================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
cd "${PROJECT_ROOT}"

require_cmd docker

# ── python:3.12-slim 베이스 이미지 로드 및 push ────────────
SLIM_TAR="offline-packages/python-3.12-slim.tar"
if [[ -f "${SLIM_TAR}" ]]; then
    log_info "python:3.12-slim 이미지 로드 중..."
    docker load -i "${SLIM_TAR}"
    docker tag python:3.12-slim "${REGISTRY}/python:3.12-slim"
    docker push "${REGISTRY}/python:3.12-slim"
    log_ok "python:3.12-slim → ${REGISTRY}/python:3.12-slim"
else
    log_info "python-3.12-slim.tar 없음, 이미 레지스트리에 있다고 가정합니다."
fi

# ── python-cpu 빌드 및 push ────────────────────────────────
log_info "python-cpu 빌드 중... (태그: ${IMAGE_TAG})"
docker build \
    --no-cache \
    -t "${PYTHON_CPU_IMAGE}" \
    -f mlops_platform/base-images/python-cpu/Dockerfile \
    .
docker push "${PYTHON_CPU_IMAGE}"
log_ok "push 완료: ${PYTHON_CPU_IMAGE}"

# ── pytorch-cpu 빌드 및 push ──────────────────────────────
log_info "pytorch-cpu 빌드 중... (태그: ${IMAGE_TAG})"
docker build \
    --no-cache \
    --build-arg IMAGE_TAG="${IMAGE_TAG}" \
    -t "${PYTORCH_CPU_IMAGE}" \
    -f mlops_platform/base-images/pytorch-cpu/Dockerfile \
    .
docker push "${PYTORCH_CPU_IMAGE}"
log_ok "push 완료: ${PYTORCH_CPU_IMAGE}"

echo ""
log_ok "플랫폼 베이스 이미지 빌드 및 push 완료."
docker images | grep -E "${REGISTRY}/(python-cpu|pytorch-cpu)"
