from pathlib import Path

from mouse_skull_pipelines.visualization.viewer import visualize_subject_interactive


def run(subject: str, reference_cohort: str) -> None:
    visualize_subject_interactive(
        subject_path=Path(subject),
        reference_cohort=Path(reference_cohort),
    )
