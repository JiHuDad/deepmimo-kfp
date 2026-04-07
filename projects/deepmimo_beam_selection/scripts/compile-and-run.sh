#!/usr/bin/env bash
# ============================================================
# compile-and-run.sh
#
# DeepMIMO 빔 선택 파이프라인을 컴파일하고 KFP에 업로드하여 실행한다.
# ============================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../mlops_platform/scripts/lib/common.sh"
cd "${PROJECT_ROOT}"

require_cmd python3
ensure_venv
start_kfp_portforward

PIPELINE_YAML="deepmimo_pipeline.yaml"
PIPELINE_NAME="${PIPELINE_NAME:-deepmimo-beam-selection}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepmimo-experiments}"
RUN_NAME="${RUN_NAME:-deepmimo-run-$(date +%Y%m%d-%H%M%S)}"

# ── 파이프라인 컴파일 ──────────────────────────────────────
log_info "파이프라인 컴파일 중..."
python3 projects/deepmimo_beam_selection/compile.py --output "${PIPELINE_YAML}"
log_ok "컴파일 완료: ${PIPELINE_YAML}"

# ── KFP에 업로드 및 실행 ──────────────────────────────────
log_info "KFP에 파이프라인 업로드 및 실행 중..."
log_info "  엔드포인트: ${KFP_ENDPOINT}"
log_info "  실행 이름: ${RUN_NAME}"

python3 - <<PYTHON
import sys
sys.path.insert(0, ".")

import kfp

client = kfp.Client(host="${KFP_ENDPOINT}")

# 실험 생성 또는 재사용
try:
    experiment = client.get_experiment(experiment_name="${EXPERIMENT_NAME}")
    print(f"기존 실험 사용: {experiment.experiment_id}")
except Exception:
    experiment = client.create_experiment(name="${EXPERIMENT_NAME}")
    print(f"새 실험 생성: {experiment.experiment_id}")

# 파이프라인 실행 (캐시 비활성화로 최신 컴포넌트 코드 강제 실행)
run = client.create_run_from_pipeline_package(
    pipeline_file="${PIPELINE_YAML}",
    arguments={
        "scenario_name": "O1_60",
        "num_epochs": 50,
        "learning_rate": 0.001,
        "batch_size": 256,
        "hidden_dims": "256,128,64",
    },
    run_name="${RUN_NAME}",
    experiment_name="${EXPERIMENT_NAME}",
    enable_caching=False,
)
print(f"실행 시작: run_id={run.run_id}")
print(f"KFP UI에서 확인: ${KFP_ENDPOINT}/#/runs/details/{run.run_id}")
PYTHON

log_ok "파이프라인 실행 완료. KFP UI에서 진행 상황을 확인하세요:"
log_ok "  http://${SERVER_IP}:31380"
