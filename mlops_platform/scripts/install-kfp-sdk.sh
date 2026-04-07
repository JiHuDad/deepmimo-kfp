#!/usr/bin/env bash
# ============================================================
# install-kfp-sdk.sh
#
# 가상환경(.venv)을 생성하고 KFP Python SDK를 오프라인 wheels에서 설치한다.
#
# 전제조건:
#   - offline-packages/wheels/ 에 kfp 관련 whl 파일이 있어야 함
# ============================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
cd "${PROJECT_ROOT}"

require_cmd python3

WHEELS_DIR="offline-packages/wheels"

if [[ ! -d "${WHEELS_DIR}" || -z "$(ls -A "${WHEELS_DIR}" 2>/dev/null)" ]]; then
    log_error "'${WHEELS_DIR}' 디렉토리가 비어있거나 존재하지 않습니다."
    log_error "먼저 'bash offline-packages/collect.sh'를 실행하세요."
    exit 1
fi

# 가상환경 생성 및 활성화
ensure_venv

log_info "KFP SDK 오프라인 설치 중... (가상환경: ${VENV_DIR})"
pip install \
    --no-index \
    --find-links="${WHEELS_DIR}" \
    kfp==2.15.0 \
    kfp-kubernetes

log_ok "설치 완료."
pip show kfp | grep -E "^(Name|Version)"
pip show kfp-kubernetes | grep -E "^(Name|Version)"
echo ""
log_info "이후 작업 시 가상환경을 활성화하세요:"
log_info "  source .venv/bin/activate"
