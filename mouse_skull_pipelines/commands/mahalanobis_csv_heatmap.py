from pathlib import Path
from typing import Optional

from mouse_skull_pipelines.analysis.cohort import create_mahalanobis_heatmap_from_csv


def run(
    mahal_csv: str,
    reference_mesh: str,
    output_dir: str,
    subject: Optional[str],
    fixate_plotting_max: Optional[float],
) -> None:
    create_mahalanobis_heatmap_from_csv(
        mahal_csv=Path(mahal_csv),
        reference_mesh=Path(reference_mesh),
        output_dir=Path(output_dir),
        subject=subject,
        fixate_plotting_max=fixate_plotting_max,
    )
