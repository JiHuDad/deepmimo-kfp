#!/usr/bin/env bash
# ============================================================
# setup-k8s.sh
#
# DeepMIMO 프로젝트용 Kubernetes PV/PVC를 생성한다.
# 시나리오 데이터는 복사하지 않고 hostPath PV로 직접 마운트.
# ============================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../mlops_platform/scripts/lib/common.sh"
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

# ── hostPath PV + PVC 생성 (범용 템플릿에서 치환) ────────────
log_info "DeepMIMO 시나리오 PV/PVC 생성 중..."
sed -e "s|DATA_HOST_PATH_PLACEHOLDER|${SCENARIO_HOST_PATH}|g" \
    -e "s|NODE_NAME_PLACEHOLDER|${NODE_NAME}|g" \
    -e "s|DATA_PV_NAME_PLACEHOLDER|deepmimo-scenarios-pv|g" \
    -e "s|DATA_PVC_NAME_PLACEHOLDER|deepmimo-scenarios|g" \
    mlops_platform/k8s/pv-data.yaml | kubectl apply -f -

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
log_ok "DeepMIMO K8s 리소스 준비 완료."
