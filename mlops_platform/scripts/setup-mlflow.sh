#!/usr/bin/env bash
# setup-mlflow.sh — MLflow ConfigMap + Secret 을 k8s 클러스터에 적용
#
# 사용법:
#   bash mlops_platform/scripts/setup-mlflow.sh
#
# 선택적 환경변수:
#   MINIO_ACCESS_KEY  — MinIO 접근 키 (기본값: minioadmin)
#   MINIO_SECRET_KEY  — MinIO 비밀 키 (기본값: minioadmin)
#   NAMESPACE         — 적용할 네임스페이스 (기본값: kubeflow)
#
# 주의: 실제 운영 환경에서는 Secret을 직접 스크립트에 하드코딩하지 마십시오.
#       SealedSecret 또는 External Secrets Operator 사용을 권장합니다.

set -euo pipefail

NAMESPACE=${NAMESPACE:-kubeflow}
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-minioadmin}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-minioadmin}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[setup-mlflow] ConfigMap 적용..."
kubectl apply -f "${REPO_ROOT}/mlops_platform/k8s/mlflow/configmap.yaml"

echo "[setup-mlflow] Secret 생성 (kubectl create secret)..."
# 기존 Secret 삭제 후 재생성 (idempotent)
kubectl delete secret mlflow-s3-creds -n "${NAMESPACE}" --ignore-not-found
kubectl create secret generic mlflow-s3-creds \
    --namespace="${NAMESPACE}" \
    --from-literal=AWS_ACCESS_KEY_ID="${MINIO_ACCESS_KEY}" \
    --from-literal=AWS_SECRET_ACCESS_KEY="${MINIO_SECRET_KEY}"

echo "[setup-mlflow] 완료. 등록된 리소스:"
kubectl get configmap mlflow-config -n "${NAMESPACE}"
kubectl get secret mlflow-s3-creds -n "${NAMESPACE}"
