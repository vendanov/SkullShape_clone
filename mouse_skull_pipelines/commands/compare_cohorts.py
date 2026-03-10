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
        )
        return

    compare_cohorts(cohort_1, cohort_2, reference_mouse, result_path)
