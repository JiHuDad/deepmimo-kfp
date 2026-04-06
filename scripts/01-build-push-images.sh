#!/usr/bin/env bash
# ============================================================
# 01-build-push-images.sh
#
# deepmimo-base 및 deepmimo-trainer Docker 이미지를 빌드하고
# 로컬 레지스트리(192.168.1.112:5000)에 push한다.
#
# 전제조건:
#   - offline-packages/wheels/ 에 whl 파일이 있어야 함
#   - offline-packages/python-3.12-slim.tar 이 있어야 함
#   - docker daemon 실행 중
# ============================================================
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
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

# ── deepmimo-base 빌드 및 push ─────────────────────────────
log_info "deepmimo-base 빌드 중... (태그: ${IMAGE_TAG})"
docker build \
    --no-cache \
    -t "${BASE_IMAGE}" \
    -f docker/base/Dockerfile \
    .
docker push "${BASE_IMAGE}"
log_ok "push 완료: ${BASE_IMAGE}"

# ── deepmimo-trainer 빌드 및 push ──────────────────────────
log_info "deepmimo-trainer 빌드 중... (태그: ${IMAGE_TAG})"
docker build \
    --no-cache \
    -t "${TRAINER_IMAGE}" \
    -f docker/trainer/Dockerfile \
    .
docker push "${TRAINER_IMAGE}"
log_ok "push 완료: ${TRAINER_IMAGE}"

echo ""
log_ok "모든 이미지 빌드 및 push 완료."
docker images | grep "${REGISTRY}/deepmimo"
