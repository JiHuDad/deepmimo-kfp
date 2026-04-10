#!/usr/bin/env bash
# ============================================================
# load-kfp-images.sh
#
# 폐쇄망 서버에서 KFP 내부 이미지를 k3s containerd 에 로드한다.
#
# 배경:
#   KFP v2 는 모든 파이프라인 스텝 Pod 에 kfp-launcher 사이드카를 주입한다.
#   이 이미지가 노드 containerd 캐시에 없으면 ImagePullBackOff 로 스텝이 실패한다.
#
#   - collect.sh (온라인)  : gcr.io/ml-pipeline/kfp-launcher 를 tar 로 저장
#   - 이 스크립트 (오프라인): tar 를 k3s containerd 에 import (레지스트리 불필요)
#
# 사용법:
#   bash mlops_platform/scripts/load-kfp-images.sh
#
# 전제조건:
#   - offline-packages/kfp-launcher-2.15.0.tar 존재
#   - k3s 설치 완료, /usr/local/bin/k3s 또는 PATH 에 존재
# ============================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
cd "${PROJECT_ROOT}"

require_cmd kubectl

KFP_VERSION="2.15.0"
KFP_LAUNCHER_IMAGE="gcr.io/ml-pipeline/kfp-launcher:${KFP_VERSION}"
KFP_LAUNCHER_TAR="offline-packages/kfp-launcher-${KFP_VERSION}.tar"

# ── k3s ctr 경로 결정 ────────────────────────────────────
if command -v k3s &>/dev/null; then
    CTR="k3s ctr"
elif command -v ctr &>/dev/null; then
    CTR="ctr"
else
    log_error "k3s 또는 ctr 명령을 찾을 수 없습니다."
    log_error "k3s 가 설치되어 있지 않거나 PATH 에 없습니다."
    exit 1
fi

# ── 이미 로드되어 있는지 확인 ────────────────────────────
if ${CTR} images ls 2>/dev/null | grep -q "kfp-launcher:${KFP_VERSION}"; then
    log_ok "kfp-launcher:${KFP_VERSION} 이미 containerd 에 존재 — 건너뜀"
else
    if [[ ! -f "${KFP_LAUNCHER_TAR}" ]]; then
        log_error "tar 파일을 찾을 수 없습니다: ${KFP_LAUNCHER_TAR}"
        log_error "온라인 머신에서 'make collect' 후 USB 복사가 완료되었는지 확인하세요."
        exit 1
    fi

    log_info "kfp-launcher:${KFP_VERSION} containerd 에 로드 중..."
    ${CTR} images import "${KFP_LAUNCHER_TAR}"
    log_ok "로드 완료: ${KFP_LAUNCHER_IMAGE}"
fi

# ── 로드 결과 확인 ────────────────────────────────────────
log_info "containerd 내 kfp-launcher 이미지:"
${CTR} images ls 2>/dev/null | grep kfp-launcher || log_error "이미지를 찾을 수 없습니다!"

# ── KFP 파이프라인 스텝이 실제 이미지를 사용하는지 확인 ──
# KFP v2 기본 launcher 이미지 경로와 일치하는지 검증
log_info "KFP launcher 설정 확인..."
CONFIGURED_IMAGE=$(kubectl get configmap -n kubeflow kfp-launcher \
    -o jsonpath='{.data.launcherImage}' 2>/dev/null || echo "")

if [[ -z "${CONFIGURED_IMAGE}" ]]; then
    # kfp-launcher ConfigMap 없음 → KFP 기본값 사용 (gcr.io/ml-pipeline/kfp-launcher:<version>)
    # containerd 에 로드된 이미지가 기본값과 일치하므로 추가 설정 불필요
    log_ok "KFP launcher ConfigMap 없음 — containerd 직접 로드로 충분합니다."
else
    log_info "KFP launcher 설정 이미지: ${CONFIGURED_IMAGE}"
    if [[ "${CONFIGURED_IMAGE}" != "${KFP_LAUNCHER_IMAGE}" ]]; then
        log_info "launcher 이미지가 다릅니다. ConfigMap 을 업데이트합니다..."
        kubectl patch configmap kfp-launcher -n kubeflow \
            --type merge \
            -p "{\"data\":{\"launcherImage\":\"${KFP_LAUNCHER_IMAGE}\"}}"
        log_ok "ConfigMap 업데이트 완료: ${KFP_LAUNCHER_IMAGE}"
    fi
fi

echo ""
log_ok "kfp-launcher 로드 완료. 파이프라인 스텝이 정상 실행됩니다."
