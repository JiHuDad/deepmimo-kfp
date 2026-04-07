"""
pipeline_helpers - 파이프라인 공통 유틸리티

프로젝트 파이프라인에서 공통으로 사용하는 헬퍼 함수 모음.
"""

from kfp import dsl, kubernetes


def mount_data_pvc(task: dsl.PipelineTask, pvc_name: str, mount_path: str):
    """데이터 PVC를 태스크에 마운트하는 헬퍼."""
    kubernetes.mount_pvc(task, pvc_name=pvc_name, mount_path=mount_path)
    return task


def set_resource_request(
    task: dsl.PipelineTask,
    cpu: str = "1",
    memory: str = "2Gi",
):
    """태스크 리소스 요청을 설정하는 헬퍼."""
    task.set_cpu_request(cpu)
    task.set_memory_request(memory)
    return task
