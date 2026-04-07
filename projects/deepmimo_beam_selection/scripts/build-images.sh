#!/usr/bin/env bash
# ============================================================
# build-images.sh
#
# DeepMIMO 프로젝트 전용 Docker 이미지를 빌드하고
# 로컬 레지스트리(localhost:5000)에 push한다.
#
# 전제조건:
#   - 플랫폼 베이스 이미지(python-cpu)가 먼저 빌드되어 있어야 함
#   - offline-packages/wheels/ 에 DeepMIMO whl 파일이 있어야 함
# ============================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../mlops_platform/scripts/lib/common.sh"
cd "${PROJECT_ROOT}"

require_cmd docker

DEEPMIMO_IMAGE="${REGISTRY}/deepmimo:${IMAGE_TAG}"

log_info "deepmimo 이미지 빌드 중... (태그: ${IMAGE_TAG})"
docker build \
    --no-cache \
    --build-arg IMAGE_TAG="${IMAGE_TAG}" \
    -t "${DEEPMIMO_IMAGE}" \
    -f projects/deepmimo_beam_selection/docker/deepmimo/Dockerfile \
    .
docker push "${DEEPMIMO_IMAGE}"
log_ok "push 완료: ${DEEPMIMO_IMAGE}"
