# MLOps 플랫폼 + DeepMIMO 프로젝트
# 사용법: make <target>

.PHONY: help collect copy-scenarios install-sdk \
        build-platform build-project build load-kfp-images \
        setup-platform setup-project setup \
        compile run all clean clean-k8s clean-k8s-force

REGISTRY     := localhost:5000
IMAGE_TAG    ?= latest

export IMAGE_TAG

help: ## 사용 가능한 명령 목록
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── 폐쇄망 준비 ───────────────────────────────────────────
collect: ## [온라인 머신] whl/이미지/시나리오 수집 (offline-packages/collect.sh)
	@bash offline-packages/collect.sh

copy-scenarios: ## USB 복사 후 시나리오를 호스트 경로에 배치
	@bash mlops_platform/scripts/copy-scenarios.sh

install-sdk: ## KFP SDK 오프라인 설치
	@bash mlops_platform/scripts/install-kfp-sdk.sh

# ── Docker 이미지 ─────────────────────────────────────────
build-platform: ## 플랫폼 베이스 이미지 빌드 (python-cpu, pytorch-cpu)
	@bash mlops_platform/scripts/build-base-images.sh

build-project: ## DeepMIMO 프로젝트 이미지 빌드 (deepmimo)
	@bash projects/deepmimo_beam_selection/scripts/build-images.sh

build: build-platform build-project ## 전체 Docker 이미지 빌드 (플랫폼 + 프로젝트)

load-kfp-images: ## KFP 내부 이미지를 k3s containerd 에 로드 (kfp-launcher 등)
	@bash mlops_platform/scripts/load-kfp-images.sh

# ── Kubernetes 설정 ───────────────────────────────────────
setup-platform: ## 플랫폼 공용 K8s 리소스 생성
	@bash mlops_platform/scripts/setup-k8s.sh

setup-project: ## DeepMIMO 프로젝트 K8s 리소스 생성 (시나리오 PV/PVC)
	@bash projects/deepmimo_beam_selection/scripts/setup-k8s.sh

setup: setup-platform setup-project ## 전체 K8s 리소스 생성

# ── 파이프라인 ────────────────────────────────────────────
compile: ## 파이프라인 YAML 컴파일
	@python3 projects/deepmimo_beam_selection/compile.py

run: ## 파이프라인 컴파일 및 KFP 실행
	@bash projects/deepmimo_beam_selection/scripts/compile-and-run.sh

# ── 전체 실행 순서 ────────────────────────────────────────
all: install-sdk build load-kfp-images setup run ## 전체 실행 (SDK 설치 → 이미지 빌드 → KFP 이미지 로드 → K8s 설정 → 파이프라인 실행)

# ── 정리 ──────────────────────────────────────────────────
clean: ## 컴파일된 YAML 및 임시 파일 정리
	@rm -f deepmimo_pipeline.yaml
	@find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "정리 완료."

clean-k8s: ## Kubernetes 리소스 전체 정리 (PV/PVC finalizer 강제 제거 포함)
	@bash mlops_platform/scripts/cleanup-k8s.sh

clean-k8s-force: ## Kubernetes 리소스 전체 정리 (확인 프롬프트 없이)
	@bash mlops_platform/scripts/cleanup-k8s.sh --force
