#!/usr/bin/env bash
# ============================================================
# 03-setup-k8s.sh
#
# Kubernetes 리소스(PVC)를 생성하고
# 시나리오 데이터를 PVC에 적재한다.
# ============================================================
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "${PROJECT_ROOT}"

require_cmd kubectl

NAMESPACE="kubeflow"

# ── PVC 생성 ──────────────────────────────────────────────
log_info "PVC 생성 중..."
kubectl apply -f k8s/pvc-scenarios.yaml -n "${NAMESPACE}"
kubectl apply -f k8s/pvc-artifacts.yaml -n "${NAMESPACE}"

# local-path 프로비저너는 WaitForFirstConsumer 모드:
# Pod가 마운트하기 전까지 Pending 상태가 정상 → Bound 대기 생략
log_info "PVC 생성 확인 중..."
for pvc in deepmimo-scenarios deepmimo-artifacts; do
    phase=$(kubectl get pvc "${pvc}" -n "${NAMESPACE}" -o jsonpath='{.status.phase}' 2>/dev/null)
    log_ok "PVC '${pvc}': ${phase:-Unknown} (Pod 마운트 시 Bound로 전환됨)"
done

# ── 시나리오 데이터 적재 ───────────────────────────────────
SCENARIO_HOST_PATH="${SCENARIO_HOST_PATH:-${HOME}/data/deepmimo-scenarios}"

if [[ ! -d "${SCENARIO_HOST_PATH}" ]]; then
    mkdir -p "${SCENARIO_HOST_PATH}"
    log_info "디렉토리 생성: ${SCENARIO_HOST_PATH}"
fi

# 시나리오 파일 유무 확인
if [[ -z "$(ls -A "${SCENARIO_HOST_PATH}" 2>/dev/null)" ]]; then
    log_info "------------------------------------------------------"
    log_info "PVC 생성 완료. 시나리오 데이터 적재는 데이터 준비 후 별도 실행:"
    log_info "  1. deepmimo.net에서 O1_60 시나리오 다운로드"
    log_info "  2. 압축 해제 후 아래 경로에 배치:"
    log_info "     ${SCENARIO_HOST_PATH}/O1_60/"
    log_info "  3. 이후 실행: make load-scenarios"
    log_info "------------------------------------------------------"
    exit 0
fi

log_info "시나리오 데이터를 PVC에 적재하는 Job 실행 중..."
kubectl delete job deepmimo-load-scenarios -n "${NAMESPACE}" --ignore-not-found
sed "s|SCENARIO_HOST_PATH_PLACEHOLDER|${SCENARIO_HOST_PATH}|g" \
    k8s/load-scenario-job.yaml | kubectl apply -f - -n "${NAMESPACE}"

log_info "Job 완료 대기 중 (최대 5분)..."
kubectl wait job/deepmimo-load-scenarios \
    -n "${NAMESPACE}" \
    --for=condition=complete \
    --timeout=300s

log_ok "시나리오 데이터 적재 완료."
kubectl logs -n "${NAMESPACE}" \
    -l job-name=deepmimo-load-scenarios \
    --tail=20
