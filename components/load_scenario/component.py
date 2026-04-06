"""
load_scenario 컴포넌트

폐쇄망 환경에서 사전 적재된 PVC로부터 DeepMIMO 시나리오 파일을 복사한다.
온라인 환경이라면 DeepMIMO.download_scenario()를 사용할 수 있지만,
여기서는 /data/scenarios PVC 마운트 경로에서 로컬 복사 방식을 사용한다.
"""

from kfp import dsl
from kfp.dsl import Output, Dataset


@dsl.component(
    base_image="localhost:5000/deepmimo-base:latest",
    packages_to_install=[],
)
def load_scenario(
    scenario_name: str,
    scenario_source_path: str,
    output_scenario: Output[Dataset],
) -> None:
    """PVC에서 시나리오 폴더를 아티팩트 경로로 복사."""
    import os
    import shutil

    src = os.path.join(scenario_source_path, scenario_name)
    dst = output_scenario.path

    if not os.path.exists(src):
        raise FileNotFoundError(
            f"시나리오 '{scenario_name}'를 '{src}'에서 찾을 수 없습니다. "
            f"PVC에 시나리오 데이터가 적재되어 있는지 확인하세요."
        )

    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)

    copied = os.listdir(dst)
    print(f"[load_scenario] '{scenario_name}' 복사 완료 → {dst}")
    print(f"[load_scenario] 파일 목록: {copied}")
    output_scenario.metadata["scenario_name"] = scenario_name
    output_scenario.metadata["file_count"] = len(copied)
