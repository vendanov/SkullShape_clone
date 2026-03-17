from pathlib import Path
from typing import Optional

from mouse_skull_pipelines.analysis.cohort import compare_cohorts, compare_cohorts_per_subject


def run(
    cohort_1: str,
    cohort_2: str,
    reference_mouse: str,
    result_dir: str,
    method: str = "group",
    ref_distribution: str = "cohort_2",
    fixate_plotting_max: Optional[float] = None,
    area_threshold: float = 5.0,
    recompute_area_percent_threshold: float = 40.0,
    recompute_face_percent_threshold: float = 40.0,
) -> None:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    if method == "per_subject":
        compare_cohorts_per_subject(
            cohort_1,
            cohort_2,
            reference_mouse,
            result_path,
            ref_distribution=ref_distribution,
            fixate_plotting_max=fixate_plotting_max,
            area_threshold=area_threshold,
            recompute_area_percent_threshold=recompute_area_percent_threshold,
            recompute_face_percent_threshold=recompute_face_percent_threshold,
        )
        return

    compare_cohorts(cohort_1, cohort_2, reference_mouse, result_path)
