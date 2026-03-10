from pathlib import Path
from typing import List, Optional

from mouse_skull_pipelines.services.workflows import prepare_full_body_and_extract_heads


def run(
    cohort_dir: str,
    ref_image_dir: Optional[str],
    cores: int,
    keep_going: bool,
    config_override: Optional[List[str]],
) -> None:
    prepare_full_body_and_extract_heads(
        cohort_dir=Path(cohort_dir),
        ref_image_dir=Path(ref_image_dir) if ref_image_dir else None,
        overrides=config_override or [],
        cores=cores,
        keep_going=keep_going,
    )
