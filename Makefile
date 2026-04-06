# DeepMIMO + Kubeflow Pipelines 프로젝트
# 사용법: make <target>

.PHONY: help build push setup compile run all clean clean-k8s

REGISTRY     := localhost:5000
IMAGE_TAG    ?= latest
KFP_UI       := http://192.168.1.112:31380
# KFP API는 ClusterIP라 port-forward 필요 → scripts/04-compile-and-run.sh에서 자동 처리

export IMAGE_TAG
export KFP_ENDPOINT

help: ## 사용 가능한 명령 목록
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── 폐쇄망 준비 ───────────────────────────────────────────
collect: ## [온라인 머신] whl/이미지/시나리오 수집 (offline-packages/collect.sh)
	@bash offline-packages/collect.sh

copy-scenarios: ## USB 복사 후 시나리오를 호스트 경로에 배치
	@bash scripts/00-copy-scenarios.sh

install-sdk: ## KFP SDK 오프라인 설치
	@bash scripts/02-install-kfp-sdk.sh

# ── Docker 이미지 ─────────────────────────────────────────
build: ## Docker 이미지 빌드 및 push
	@bash scripts/01-build-push-images.sh

# ── Kubernetes 설정 ───────────────────────────────────────
setup: ## hostPath PV/PVC 생성 (시나리오 데이터는 PVC에 직접 마운트됨, 복사 없음)
	@bash scripts/03-setup-k8s.sh

# ── 파이프라인 ────────────────────────────────────────────
compile: ## 파이프라인 YAML 컴파일
	@python3 pipelines/compile.py

run: ## 파이프라인 컴파일 및 KFP 실행
	@bash scripts/04-compile-and-run.sh

# ── 전체 실행 순서 ────────────────────────────────────────
all: install-sdk build setup run ## 전체 실행 (SDK 설치 → 이미지 빌드 → hostPath PVC 생성 → 파이프라인 실행)

# ── 정리 ──────────────────────────────────────────────────
clean: ## 컴파일된 YAML 및 임시 파일 정리
	@rm -f deepmimo_pipeline.yaml
	@find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "정리 완료."

clean-k8s: ## Kubernetes PV/PVC 삭제 (시나리오 원본 데이터는 삭제되지 않음)
	@kubectl delete pvc deepmimo-scenarios deepmimo-artifacts -n kubeflow --ignore-not-found
	@kubectl delete pv deepmimo-scenarios-pv --ignore-not-found
	@echo "K8s 리소스 삭제 완료."
