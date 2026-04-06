#!/usr/bin/env bash
# 공통 변수 및 함수

# 서버 IP (실제 IP로 수정)
SERVER_IP="192.168.1.112"
REGISTRY="${SERVER_IP}:5000"
KFP_ENDPOINT="http://${SERVER_IP}:31380"

# 이미지 태그
IMAGE_TAG="${IMAGE_TAG:-latest}"

BASE_IMAGE="${REGISTRY}/deepmimo-base:${IMAGE_TAG}"
TRAINER_IMAGE="${REGISTRY}/deepmimo-trainer:${IMAGE_TAG}"

# 프로젝트 루트 (scripts/lib 기준 두 단계 위)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

log_info()  { echo "[INFO]  $*"; }
log_ok()    { echo "[OK]    $*"; }
log_error() { echo "[ERROR] $*" >&2; }

require_cmd() {
    command -v "$1" &>/dev/null || { log_error "'$1' 명령을 찾을 수 없습니다."; exit 1; }
}
