#!/usr/bin/env bash
# ============================================================
# 03-setup-k8s.sh
#
# Kubernetes PV/PVC를 생성한다.
# 시나리오 데이터는 복사하지 않고 hostPath PV로 직접 마운트.
# (복사 시 21GB 중복 발생 방지)
# ============================================================
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "${PROJECT_ROOT}"

require_cmd kubectl

NAMESPACE="kubeflow"
SCENARIO_HOST_PATH="${SCENARIO_HOST_PATH:-${HOME}/data/deepmimo-scenarios}"
NODE_NAME=$(kubectl get node -o jsonpath='{.items[0].metadata.name}')

if [[ ! -d "${SCENARIO_HOST_PATH}" ]]; then
    log_error "시나리오 경로가 없습니다: ${SCENARIO_HOST_PATH}"
    log_error "먼저 'make copy-scenarios' 를 실행하세요."
    exit 1
fi

log_info "시나리오 경로: ${SCENARIO_HOST_PATH} ($(du -sh "${SCENARIO_HOST_PATH}" | cut -f1))"
log_info "노드: ${NODE_NAME}"

# ── hostPath PV + PVC 생성 (데이터 복사 없음) ─────────────
log_info "hostPath PV/PVC 생성 중..."
sed -e "s|SCENARIO_HOST_PATH_PLACEHOLDER|${SCENARIO_HOST_PATH}|g" \
    -e "s|leaf007|${NODE_NAME}|g" \
    k8s/pv-scenarios.yaml | kubectl apply -f -

# ── artifacts PVC 생성 ────────────────────────────────────
kubectl apply -f k8s/pvc-artifacts.yaml -n "${NAMESPACE}"

# ── PVC Bound 확인 ────────────────────────────────────────
log_info "PVC Bound 대기 중..."
for i in $(seq 1 15); do
    phase=$(kubectl get pvc deepmimo-scenarios -n "${NAMESPACE}" \
        -o jsonpath='{.status.phase}' 2>/dev/null)
    if [[ "${phase}" == "Bound" ]]; then
        log_ok "deepmimo-scenarios: Bound"
        break
    fi
    sleep 2
done

kubectl get pvc -n "${NAMESPACE}" | grep deepmimo
log_ok "K8s 리소스 준비 완료."
