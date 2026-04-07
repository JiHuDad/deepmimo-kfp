"""
파이프라인 YAML 컴파일 스크립트.

사용법:
    python projects/deepmimo_beam_selection/compile.py
    python projects/deepmimo_beam_selection/compile.py --output my_pipeline.yaml
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
_project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, _project_root)

from kfp import compiler


def main():
    parser = argparse.ArgumentParser(description="DeepMIMO 빔 선택 파이프라인 컴파일")
    parser.add_argument(
        "--output",
        default="deepmimo_pipeline.yaml",
        help="출력 YAML 파일 경로 (기본값: deepmimo_pipeline.yaml)",
    )
    args = parser.parse_args()

    from projects.deepmimo_beam_selection.pipeline import deepmimo_pipeline

    print(f"파이프라인 컴파일 중... → {args.output}")
    compiler.Compiler().compile(
        pipeline_func=deepmimo_pipeline,
        package_path=args.output,
    )
    print(f"완료: {args.output}")


if __name__ == "__main__":
    main()
