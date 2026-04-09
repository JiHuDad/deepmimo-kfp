"""
validate_data 컴포넌트 (범용)

PVC에 마운트된 데이터 경로를 검증하고 절대 경로를 아티팩트로 전달한다.
데이터를 복사하지 않으므로 디스크를 낭비하지 않는다.

사용 예:
    validate_task = validate_data(data_name="asu_campus_3p5", data_source_path="/data/scenarios")
    kubernetes.mount_pvc(validate_task, pvc_name="my-data-pvc", mount_path="/data/scenarios")
"""

import os

from kfp import dsl
from kfp.dsl import Output, Dataset

_IMAGE_TAG = os.environ.get("IMAGE_TAG", "latest")


@dsl.component(
    base_image=f"localhost:5000/python-cpu:{_IMAGE_TAG}",
    packages_to_install=[],
)
def validate_data(
    data_name: str,
    data_source_path: str,
    output_dataset: Output[Dataset],
) -> None:
    """PVC 마운트 경로에서 데이터 존재를 확인하고 절대 경로를 아티팩트에 기록."""
    import os

    src = os.path.join(data_source_path, data_name)

    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"데이터 '{data_name}'를 '{src}'에서 찾을 수 없습니다. "
            f"PVC에 데이터가 적재되어 있는지 확인하세요."
        )

    file_count = len(os.listdir(src))
    print(f"[validate_data] '{data_name}' 확인 완료: {src} ({file_count}개 파일)")

    # 절대 경로를 텍스트 파일로 저장 (데이터 복사 없음)
    os.makedirs(os.path.dirname(output_dataset.path), exist_ok=True)
    with open(output_dataset.path, "w") as f:
        f.write(src)

    output_dataset.metadata["data_name"] = data_name
    output_dataset.metadata["data_path"] = src
    output_dataset.metadata["file_count"] = file_count
