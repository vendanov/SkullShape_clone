from pathlib import Path
from typing import Optional

from mouse_skull_pipelines.analysis.cohort import one_cohort_statistics


def run(
    cohort: str,
    reference_mouse: str,
    result_dir: str,
    upper_limit: Optional[float],
    mahalanobis_summary_csv: Optional[str],
    mahal_threshold: float,
    skip_variability_gif: bool,
    fast_outliers: bool,
) -> None:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    one_cohort_statistics(
        cohort,
        reference_mouse,
        result_path,
        upper_limit=upper_limit,
        mahal_summary_csv=mahalanobis_summary_csv,
        mahal_threshold=mahal_threshold,
        skip_variability_gif=skip_variability_gif,
        fast_outliers=fast_outliers,
    )
