"""
load_scenario 컴포넌트

폐쇄망 환경에서 사전 적재된 PVC의 시나리오 경로를 검증하고
절대 경로를 아티팩트로 전달한다. 데이터를 복사하지 않으므로 디스크를 낭비하지 않는다.
"""

import os

from kfp import dsl
from kfp.dsl import Output, Dataset

_IMAGE_TAG = os.environ.get("IMAGE_TAG", "latest")

@dsl.component(
    base_image=f"localhost:5000/deepmimo-base:{_IMAGE_TAG}",
    packages_to_install=[],
)
def load_scenario(
    scenario_name: str,
    scenario_source_path: str,
    output_scenario: Output[Dataset],
) -> None:
    """PVC 마운트 경로에서 시나리오 존재를 확인하고 절대 경로를 아티팩트에 기록."""
    import os

    src = os.path.join(scenario_source_path, scenario_name)

    if not os.path.isdir(src):
        raise FileNotFoundError(
            f"시나리오 '{scenario_name}'를 '{src}'에서 찾을 수 없습니다. "
            f"PVC에 시나리오 데이터가 적재되어 있는지 확인하세요."
        )

    file_count = len(os.listdir(src))
    print(f"[load_scenario] '{scenario_name}' 확인 완료: {src} ({file_count}개 파일)")

    # 절대 경로를 텍스트 파일로 저장 (데이터 복사 없음)
    os.makedirs(os.path.dirname(output_scenario.path), exist_ok=True)
    with open(output_scenario.path, "w") as f:
        f.write(src)

    output_scenario.metadata["scenario_name"] = scenario_name
    output_scenario.metadata["scenario_path"] = src
    output_scenario.metadata["file_count"] = file_count
