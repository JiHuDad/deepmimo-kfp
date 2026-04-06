#!/usr/bin/env bash
# ============================================================
# collect.sh — 온라인 머신에서 실행하는 오프라인 패키지 수집 스크립트
#
# 사용법 (인터넷이 되는 머신에서):
#   chmod +x collect.sh && ./collect.sh
#
# 결과물을 USB에 복사 후 폐쇄망 서버로 이동:
#   offline-packages/wheels/    ← pip whl 파일들
#   offline-packages/python-3.12-slim.tar
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHEELS_DIR="${SCRIPT_DIR}/wheels"
VENV_DIR="${PROJECT_ROOT}/.venv"
mkdir -p "${WHEELS_DIR}"

# Ubuntu 24.04은 시스템 pip 사용 불가(PEP 668).
# 가상환경을 생성하고 그 안의 pip로 download한다.
if [[ ! -f "${VENV_DIR}/bin/pip" ]]; then
    echo "[INFO] 가상환경 생성 중: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi
PIP="${VENV_DIR}/bin/pip"
echo "[INFO] pip: $("${PIP}" --version)"

echo "=== [1/3] Python 패키지 (whl) 수집 ==="

# [A0] 빌드 도구 (kfp-server-api 같은 sdist 빌드에 필요)
echo "[A0] setuptools / wheel (sdist 빌드 도구)"
"${PIP}" download \
    "setuptools>=40.8.0" \
    wheel \
    --dest "${WHEELS_DIR}" \
    --prefer-binary

# [A] KFP SDK + DeepMIMO
echo "[A] KFP SDK + DeepMIMO"
"${PIP}" download \
    kfp==2.15.0 \
    kfp-kubernetes \
    "DeepMIMO==4.0.0" \
    --dest "${WHEELS_DIR}" \
    --prefer-binary

# [B] 과학 계산 스택 (numpy, scipy 등 C 확장 포함)
echo "[B] numpy / scipy / matplotlib / h5py / scikit-learn"
"${PIP}" download \
    "numpy>=1.24,<2.0" \
    "scipy>=1.11" \
    "matplotlib>=3.7" \
    "h5py>=3.9" \
    "scikit-learn>=1.3" \
    --dest "${WHEELS_DIR}" \
    --prefer-binary

# [C] PyTorch CPU-only (용량 주의: ~700MB)
echo "[C] PyTorch CPU"
"${PIP}" download \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --dest "${WHEELS_DIR}" \
    --index-url https://download.pytorch.org/whl/cpu \
    --prefer-binary

echo "whl 파일 수: $(ls "${WHEELS_DIR}" | wc -l)"

echo ""
echo "=== [2/3] Docker 베이스 이미지 저장 ==="
docker pull python:3.12-slim
docker save python:3.12-slim -o "${SCRIPT_DIR}/python-3.12-slim.tar"
echo "저장 완료: python-3.12-slim.tar ($(du -sh "${SCRIPT_DIR}/python-3.12-slim.tar" | cut -f1))"

echo ""
echo "=== [3/3] DeepMIMO 시나리오 데이터 다운로드 ==="

SCENARIOS_DIR="${SCRIPT_DIR}/scenarios"
SCENARIOS="${DEEPMIMO_SCENARIOS:-O1_60}"   # 쉼표 구분으로 여러 개 지정 가능

mkdir -p "${SCENARIOS_DIR}"

# DeepMIMO 설치 후 dm.download() 사용
PYTHON="${VENV_DIR}/bin/python"
"${PIP}" install --quiet "DeepMIMO==4.0.0" --find-links="${WHEELS_DIR}" --prefer-binary

IFS=',' read -ra SCENARIO_LIST <<< "${SCENARIOS}"
for scenario in "${SCENARIO_LIST[@]}"; do
    scenario="$(echo "${scenario}" | tr -d ' ')"
    if [[ -d "${SCENARIOS_DIR}/${scenario}" ]]; then
        echo "[INFO] '${scenario}' 이미 존재, 건너뜀"
        continue
    fi
    echo "[INFO] '${scenario}' 다운로드 중..."
    "${PYTHON}" - <<PYEOF
import deepmimo as dm
import os
dm.download('${scenario}', destination='${SCENARIOS_DIR}')
print(f"[OK] ${scenario} 다운로드 완료 → ${SCENARIOS_DIR}/${scenario}")
PYEOF
done

echo "시나리오 목록: $(ls "${SCENARIOS_DIR}")"

echo ""
echo "=== 수집 완료 ==="
echo "USB에 복사할 파일:"
echo "  - offline-packages/wheels/    ($(du -sh "${WHEELS_DIR}" | cut -f1))"
echo "  - offline-packages/python-3.12-slim.tar"
echo "  - offline-packages/scenarios/ ($(du -sh "${SCENARIOS_DIR}" | cut -f1))"
