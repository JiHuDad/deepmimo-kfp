#!/usr/bin/env bash
# ============================================================
# cleanup-k8s.sh
#
# 플랫폼 및 프로젝트 Kubernetes 리소스를 모두 제거한다.
#
# PVC/PV 가 Pod 에서 사용 중이면 finalizer 가 남아 삭제가 블록된다.
# 이 스크립트는 다음 순서로 안전하게 정리한다:
#   1. KFP 파이프라인 실행(Run) 종료
#   2. 관련 Pod 종료 대기
#   3. PVC finalizer 강제 제거 후 삭제
#   4. PV 삭제
#   5. ConfigMap / Secret 삭제
#
# 사용법:
#   bash mlops_platform/scripts/cleanup-k8s.sh [--force]
#
#   --force : 확인 프롬프트 없이 즉시 실행
# ============================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

require_cmd kubectl

NAMESPACE="kubeflow"
FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

# ── 확인 프롬프트 ────────────────────────────────────────
if [[ "${FORCE}" == false ]]; then
    echo ""
    echo "  이 작업은 다음 K8s 리소스를 모두 삭제합니다:"
    echo "    • PVC : mlops-artifacts, deepmimo-scenarios"
    echo "    • PV  : deepmimo-scenarios-pv"
    echo "    • ConfigMap : mlflow-config"
    echo "    • Secret    : mlflow-s3-creds"
    echo ""
    read -r -p "  계속하시겠습니까? [y/N] " answer
    [[ "${answer}" =~ ^[Yy]$ ]] || { echo "취소됨."; exit 0; }
    echo ""
fi

# ── Helper: PVC finalizer 강제 제거 ─────────────────────
# PVC 가 Terminating 상태로 멈추는 경우 kubernetes finalizer 를 제거해 강제 삭제.
remove_pvc_finalizer() {
    local pvc_name="$1"
    local phase
    phase=$(kubectl get pvc "${pvc_name}" -n "${NAMESPACE}" \
        -o jsonpath='{.status.phase}' 2>/dev/null || echo "")

    if [[ -z "${phase}" ]]; then
        return 0   # 이미 없음
    fi

    # Terminating 상태이거나 강제 모드면 finalizer 제거
    local finalizers
    finalizers=$(kubectl get pvc "${pvc_name}" -n "${NAMESPACE}" \
        -o jsonpath='{.metadata.finalizers}' 2>/dev/null || echo "")

    if [[ -n "${finalizers}" && "${finalizers}" != "[]" ]]; then
        log_info "PVC ${pvc_name} finalizer 제거 중..."
        kubectl patch pvc "${pvc_name}" -n "${NAMESPACE}" \
            -p '{"metadata":{"finalizers":[]}}' \
            --type=merge 2>/dev/null || true
    fi
}

# ── Helper: PV finalizer 강제 제거 ──────────────────────
remove_pv_finalizer() {
    local pv_name="$1"
    local exists
    exists=$(kubectl get pv "${pv_name}" --ignore-not-found \
        -o jsonpath='{.metadata.name}' 2>/dev/null || echo "")

    if [[ -z "${exists}" ]]; then
        return 0
    fi

    local finalizers
    finalizers=$(kubectl get pv "${pv_name}" \
        -o jsonpath='{.metadata.finalizers}' 2>/dev/null || echo "")

    if [[ -n "${finalizers}" && "${finalizers}" != "[]" ]]; then
        log_info "PV ${pv_name} finalizer 제거 중..."
        kubectl patch pv "${pv_name}" \
            -p '{"metadata":{"finalizers":[]}}' \
            --type=merge 2>/dev/null || true
    fi
}

# ── Helper: Pod 종료 대기 ────────────────────────────────
wait_pods_gone() {
    local label_selector="$1"
    local timeout="${2:-60}"
    local elapsed=0

    log_info "Pod 종료 대기 중 (selector: ${label_selector}, 최대 ${timeout}s)..."
    while kubectl get pods -n "${NAMESPACE}" -l "${label_selector}" \
            --ignore-not-found -o name 2>/dev/null | grep -q .; do
        sleep 3
        elapsed=$((elapsed + 3))
        if [[ ${elapsed} -ge ${timeout} ]]; then
            log_info "Pod 아직 남아 있음 — finalizer 강제 제거 후 계속 진행"
            break
        fi
    done
}

# ════════════════════════════════════════════════════════════
# STEP 1: KFP 실행 중인 파이프라인 Pod 종료
# ════════════════════════════════════════════════════════════
log_info "=== STEP 1: 실행 중인 KFP workflow Pod 종료 ==="
# Argo/KFP workflow pod 는 workflow 라벨을 가짐
kubectl delete pods -n "${NAMESPACE}" \
    -l "pipelines.kubeflow.org/pipeline-sdk-type=kfp" \
    --ignore-not-found --grace-period=10 2>/dev/null || true

# workflow 라벨이 다를 수 있으므로 Completed/Failed 외 Running Pod 도 정리
kubectl delete pods -n "${NAMESPACE}" \
    --field-selector="status.phase=Running" \
    -l "workflows.argoproj.io/workflow" \
    --ignore-not-found --grace-period=10 2>/dev/null || true

# ════════════════════════════════════════════════════════════
# STEP 2: PVC 삭제 (사용 중인 경우 finalizer 강제 제거)
# ════════════════════════════════════════════════════════════
log_info "=== STEP 2: PVC 삭제 ==="

for PVC_NAME in mlops-artifacts deepmimo-scenarios; do
    if ! kubectl get pvc "${PVC_NAME}" -n "${NAMESPACE}" \
            --ignore-not-found -o name 2>/dev/null | grep -q .; then
        log_info "PVC ${PVC_NAME}: 없음 (건너뜀)"
        continue
    fi

    log_info "PVC ${PVC_NAME} 삭제 요청..."
    kubectl delete pvc "${PVC_NAME}" -n "${NAMESPACE}" \
        --ignore-not-found --grace-period=5 2>/dev/null || true

    # 3초 후에도 남아 있으면 finalizer 강제 제거
    sleep 3
    if kubectl get pvc "${PVC_NAME}" -n "${NAMESPACE}" \
            --ignore-not-found -o name 2>/dev/null | grep -q .; then
        log_info "PVC ${PVC_NAME} Terminating 감지 — finalizer 강제 제거"
        remove_pvc_finalizer "${PVC_NAME}"
        # finalizer 제거 후 재삭제 시도
        kubectl delete pvc "${PVC_NAME}" -n "${NAMESPACE}" \
            --ignore-not-found --grace-period=0 --force 2>/dev/null || true
    fi
    log_ok "PVC ${PVC_NAME} 삭제 완료"
done

# ════════════════════════════════════════════════════════════
# STEP 3: PV 삭제
# ════════════════════════════════════════════════════════════
log_info "=== STEP 3: PV 삭제 ==="

for PV_NAME in deepmimo-scenarios-pv; do
    if ! kubectl get pv "${PV_NAME}" --ignore-not-found \
            -o name 2>/dev/null | grep -q .; then
        log_info "PV ${PV_NAME}: 없음 (건너뜀)"
        continue
    fi

    log_info "PV ${PV_NAME} 삭제 요청..."
    kubectl delete pv "${PV_NAME}" \
        --ignore-not-found --grace-period=5 2>/dev/null || true

    sleep 3
    if kubectl get pv "${PV_NAME}" --ignore-not-found \
            -o name 2>/dev/null | grep -q .; then
        log_info "PV ${PV_NAME} Terminating 감지 — finalizer 강제 제거"
        remove_pv_finalizer "${PV_NAME}"
        kubectl delete pv "${PV_NAME}" \
            --ignore-not-found --grace-period=0 --force 2>/dev/null || true
    fi
    log_ok "PV ${PV_NAME} 삭제 완료"
done

# ════════════════════════════════════════════════════════════
# STEP 4: ConfigMap / Secret 삭제
# ════════════════════════════════════════════════════════════
log_info "=== STEP 4: ConfigMap / Secret 삭제 ==="
kubectl delete configmap mlflow-config \
    -n "${NAMESPACE}" --ignore-not-found
kubectl delete secret mlflow-s3-creds \
    -n "${NAMESPACE}" --ignore-not-found
log_ok "ConfigMap / Secret 삭제 완료"

# ════════════════════════════════════════════════════════════
# 최종 상태 확인
# ════════════════════════════════════════════════════════════
echo ""
log_ok "=== 정리 완료 ==="
echo ""
echo "  남은 PVC (kubeflow):"
kubectl get pvc -n "${NAMESPACE}" 2>/dev/null || echo "  (없음)"
echo ""
echo "  남은 PV:"
kubectl get pv 2>/dev/null | grep -E "deepmimo|mlops" || echo "  (없음)"
