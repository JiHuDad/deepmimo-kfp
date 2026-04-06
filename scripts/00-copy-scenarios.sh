#!/usr/bin/env bash
# ============================================================
# 00-copy-scenarios.sh
#
# USB에서 복사해온 시나리오 데이터를 호스트 경로에 배치한다.
# collect.sh 실행 후 offline-packages/scenarios/ 에 있는
# 시나리오 폴더를 SCENARIO_HOST_PATH로 복사.
#
# 사용법:
#   bash scripts/00-copy-scenarios.sh
#   SCENARIO_HOST_PATH=/custom/path bash scripts/00-copy-scenarios.sh
# ============================================================
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "${PROJECT_ROOT}"

SCENARIO_HOST_PATH="${SCENARIO_HOST_PATH:-${HOME}/data/deepmimo-scenarios}"
SCENARIOS_SRC="${PROJECT_ROOT}/offline-packages/scenarios"

if [[ ! -d "${SCENARIOS_SRC}" ]]; then
    log_error "'${SCENARIOS_SRC}' 디렉토리가 없습니다."
    log_error "온라인 머신에서 collect.sh를 먼저 실행하고 USB로 복사하세요."
    exit 1
fi

mkdir -p "${SCENARIO_HOST_PATH}"

log_info "시나리오 복사: ${SCENARIOS_SRC} → ${SCENARIO_HOST_PATH}"
cp -rv "${SCENARIOS_SRC}/." "${SCENARIO_HOST_PATH}/"

log_ok "완료. 시나리오 목록:"
ls "${SCENARIO_HOST_PATH}"
