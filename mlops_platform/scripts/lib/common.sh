#!/usr/bin/env bash
# 공통 변수 및 함수 (플랫폼 공용)

# 서버 IP (실제 IP로 수정)
SERVER_IP="192.168.1.112"
REGISTRY="localhost:5000"
# ml-pipeline은 ClusterIP(8888)라 외부 직접 접근 불가 → port-forward 사용
KFP_ENDPOINT="http://localhost:8888"
KFP_PF_PORT=8888

# 이미지 태그
IMAGE_TAG="${IMAGE_TAG:-latest}"

# 플랫폼 베이스 이미지
PYTHON_CPU_IMAGE="${REGISTRY}/python-cpu:${IMAGE_TAG}"
PYTORCH_CPU_IMAGE="${REGISTRY}/pytorch-cpu:${IMAGE_TAG}"

# 프로젝트 루트 (scripts/lib 기준 세 단계 위: mlops_platform/scripts/lib → 루트)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

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

# ml-pipeline port-forward 시작 (백그라운드), 스크립트 종료 시 자동 정리
start_kfp_portforward() {
    if curl -s --max-time 1 "${KFP_ENDPOINT}/apis/v2beta1/healthz" &>/dev/null; then
        log_info "KFP API 이미 접근 가능 (${KFP_ENDPOINT})"
        return 0
    fi
    log_info "KFP port-forward 시작: svc/ml-pipeline ${KFP_PF_PORT}:8888"
    kubectl port-forward -n kubeflow svc/ml-pipeline \
        "${KFP_PF_PORT}:8888" --address=127.0.0.1 \
        >/tmp/kfp-portforward.log 2>&1 &
    KFP_PF_PID=$!
    trap 'kill ${KFP_PF_PID} 2>/dev/null' EXIT
    # 준비 대기 (최대 10초)
    for i in $(seq 1 10); do
        sleep 1
        if curl -s --max-time 1 "${KFP_ENDPOINT}/apis/v2beta1/healthz" &>/dev/null; then
            log_ok "KFP API 연결됨 (${KFP_ENDPOINT})"
            return 0
        fi
    done
    log_error "KFP port-forward 실패. 로그: /tmp/kfp-portforward.log"
    exit 1
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
