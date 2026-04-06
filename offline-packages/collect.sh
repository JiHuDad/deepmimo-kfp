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
WHEELS_DIR="${SCRIPT_DIR}/wheels"
mkdir -p "${WHEELS_DIR}"

echo "=== [1/3] Python 패키지 (whl) 수집 ==="
# KFP SDK (서버의 KFP 버전과 일치시킬 것)
pip download kfp==2.15.0 kfp-kubernetes \
    --dest "${WHEELS_DIR}" \
    --python-version 3.12 \
    --platform linux_x86_64 \
    --only-binary :all:

# DeepMIMO 및 과학 계산 스택
pip download \
    "DeepMIMO==3.0.0" \
    "numpy>=1.24,<2.0" \
    "scipy>=1.11" \
    "matplotlib>=3.7" \
    "h5py>=3.9" \
    "scikit-learn>=1.3" \
    --dest "${WHEELS_DIR}" \
    --python-version 3.12 \
    --platform linux_x86_64 \
    --only-binary :all:

# PyTorch CPU-only (용량 주의: ~700MB)
pip download \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --dest "${WHEELS_DIR}" \
    --index-url https://download.pytorch.org/whl/cpu \
    --python-version 3.12 \
    --platform linux_x86_64 \
    --only-binary :all:

echo "whl 파일 수: $(ls "${WHEELS_DIR}" | wc -l)"

echo ""
echo "=== [2/3] Docker 베이스 이미지 저장 ==="
docker pull python:3.12-slim
docker save python:3.12-slim -o "${SCRIPT_DIR}/python-3.12-slim.tar"
echo "저장 완료: python-3.12-slim.tar ($(du -sh "${SCRIPT_DIR}/python-3.12-slim.tar" | cut -f1))"

echo ""
echo "=== [3/3] DeepMIMO 시나리오 데이터 안내 ==="
cat <<'EOF'
  deepmimo.net 에서 O1_60 시나리오를 수동으로 다운로드하세요:
  - URL: https://deepmimo.net/scenarios/o1-scenario/
  - 다운로드 후 압축 해제하여 USB에 포함
  - 폐쇄망 서버의 /home/fall/data/deepmimo-scenarios/O1_60/ 에 배치
EOF

echo ""
echo "=== 수집 완료 ==="
echo "USB에 복사할 파일:"
echo "  - offline-packages/wheels/ ($(du -sh "${WHEELS_DIR}" | cut -f1))"
echo "  - offline-packages/python-3.12-slim.tar"
echo "  - (수동) DeepMIMO O1_60 시나리오 폴더"
