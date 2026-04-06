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

# 가상환경 경로
VENV_DIR="${PROJECT_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

log_info()  { echo "[INFO]  $*"; }
log_ok()    { echo "[OK]    $*"; }
log_error() { echo "[ERROR] $*" >&2; }

require_cmd() {
    command -v "$1" &>/dev/null || { log_error "'$1' 명령을 찾을 수 없습니다."; exit 1; }
}

# 가상환경이 없으면 생성
ensure_venv() {
    if [[ ! -f "${VENV_PYTHON}" ]]; then
        log_info "가상환경 생성 중: ${VENV_DIR}"
        python3 -m venv "${VENV_DIR}"
        log_ok "가상환경 생성 완료"
    fi
    # 현재 셸에 activate
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
}
