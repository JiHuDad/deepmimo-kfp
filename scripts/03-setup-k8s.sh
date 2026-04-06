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

# PVC Ready 대기 (최대 60초)
log_info "PVC 준비 대기 중..."
for pvc in deepmimo-scenarios deepmimo-artifacts; do
    kubectl wait pvc "${pvc}" \
        -n "${NAMESPACE}" \
        --for=jsonpath='{.status.phase}'=Bound \
        --timeout=60s && log_ok "PVC '${pvc}' Bound"
done

# ── 시나리오 데이터 적재 ───────────────────────────────────
SCENARIO_HOST_PATH="/home/fall/data/deepmimo-scenarios"
if [[ ! -d "${SCENARIO_HOST_PATH}" ]]; then
    log_error "시나리오 데이터 경로가 없습니다: ${SCENARIO_HOST_PATH}"
    log_error "deepmimo.net에서 O1_60 시나리오를 다운로드하여 아래 경로에 배치하세요:"
    log_error "  ${SCENARIO_HOST_PATH}/O1_60/"
    exit 1
fi

log_info "시나리오 데이터를 PVC에 적재하는 Job 실행 중..."
kubectl apply -f k8s/load-scenario-job.yaml -n "${NAMESPACE}"

log_info "Job 완료 대기 중 (최대 5분)..."
kubectl wait job/deepmimo-load-scenarios \
    -n "${NAMESPACE}" \
    --for=condition=complete \
    --timeout=300s

log_ok "시나리오 데이터 적재 완료."
kubectl logs -n "${NAMESPACE}" \
    -l job-name=deepmimo-load-scenarios \
    --tail=20
