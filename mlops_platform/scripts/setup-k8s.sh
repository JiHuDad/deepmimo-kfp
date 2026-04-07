#!/usr/bin/env bash
# ============================================================
# setup-k8s.sh
#
# 플랫폼 공용 Kubernetes 리소스를 생성한다.
# (artifacts PVC 등)
#
# 프로젝트별 데이터 PV/PVC는 각 프로젝트의 setup 스크립트에서 생성.
# ============================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
cd "${PROJECT_ROOT}"

require_cmd kubectl

NAMESPACE="kubeflow"

# ── artifacts PVC 생성 ────────────────────────────────────
log_info "플랫폼 공용 artifacts PVC 생성 중..."
kubectl apply -f mlops_platform/k8s/pvc-artifacts.yaml -n "${NAMESPACE}"

log_ok "플랫폼 K8s 리소스 준비 완료."
kubectl get pvc -n "${NAMESPACE}" | grep mlops
