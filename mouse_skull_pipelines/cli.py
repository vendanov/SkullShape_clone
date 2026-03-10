from typing import List, Optional

import typer

from mouse_skull_pipelines.commands import (
    compare_cohorts as compare_cohorts_cmd,
    generate_full_body as generate_full_body_cmd,
    mahalanobis_csv_heatmap as mahal_heatmap_cmd,
    one_cohort_statistics as one_cohort_stats_cmd,
    visualize_subject as visualize_subject_cmd,
)
from mouse_skull_pipelines.config.runtime import _get_app_version

app = typer.Typer(add_completion=False, help="Run the full body and skull pipeline.")


@app.command("compare_cohorts")
def compare_cohorts_command(
    cohort_1: str = typer.Option(..., "--cohort_1", help="Path to the cohort directory."),
    cohort_2: str = typer.Option(..., "--cohort_2", help="Path to the cohort directory."),
    reference_mouse: str = typer.Option(..., "--reference_mouse", help="Path to the reference mouse directory."),
    result_dir: str = typer.Option(..., "--result_dir", help="Path to the result file."),
    method: str = typer.Option("group", "--method", help="Comparison method: group (default) or per_subject"),
    ref_distribution: str = typer.Option("cohort_2", "--ref_distribution", help="For per_subject: which cohort defines the reference distribution."),
    fixate_plotting_max: Optional[float] = typer.Option(None, "--fixate_plotting_max", help="For per_subject: cap Mahalanobis distances and colorbar at this value."),
) -> None:
    if method not in {"group", "per_subject"}:
        raise typer.BadParameter("method must be one of: group, per_subject")
    if ref_distribution not in {"cohort_1", "cohort_2"}:
        raise typer.BadParameter("ref_distribution must be one of: cohort_1, cohort_2")
    compare_cohorts_cmd.run(
        cohort_1=cohort_1,
        cohort_2=cohort_2,
        reference_mouse=reference_mouse,
        result_dir=result_dir,
        method=method,
        ref_distribution=ref_distribution,
        fixate_plotting_max=fixate_plotting_max,
    )


@app.command("generate_full_body")
def generate_full_body_command(
    cohort_dir: str = typer.Option(..., "--cohort-dir", "--cohort_dir", help="Path to the cohort directory containing raw full body scans."),
    ref_image_dir: Optional[str] = typer.Option(None, "--ref-image-dir", "--ref_image_dir", help="Override the reference image directory in the config."),
    cores: int = typer.Option(8, "--cores", help="Number of CPU cores to pass to Snakemake."),
    keep_going: bool = typer.Option(False, "--keep-going", help="Pass --keep-going to Snakemake."),
    config_override: Optional[List[str]] = typer.Option(None, "--config-override", help="Override additional config entries (repeatable)."),
) -> None:
    generate_full_body_cmd.run(
        cohort_dir=cohort_dir,
        ref_image_dir=ref_image_dir,
        cores=cores,
        keep_going=keep_going,
        config_override=config_override,
    )


@app.command("visualize_subject")
def visualize_subject_command(subject: str, reference_cohort: str) -> None:
    visualize_subject_cmd.run(subject=subject, reference_cohort=reference_cohort)


@app.command("mahalanobis_csv_heatmap")
def mahalanobis_csv_heatmap_command(
    mahal_csv: str = typer.Option(..., "--mahal_csv", help="Path to CSV with subject/subj_id, face_index/face_id, mahalanobis."),
    reference_mesh: str = typer.Option(..., "--reference_mesh", help="Path to reference OBJ mesh."),
    output_dir: str = typer.Option(..., "--output_dir", help="Output directory for generated GIF(s)."),
    subject: Optional[str] = typer.Option(None, "--subject", help="Subject name to render. If omitted, render all subjects in CSV."),
    fixate_plotting_max: Optional[float] = typer.Option(None, "--fixate_plotting_max", help="Optional fixed max value for color scale."),
) -> None:
    mahal_heatmap_cmd.run(
        mahal_csv=mahal_csv,
        reference_mesh=reference_mesh,
        output_dir=output_dir,
        subject=subject,
        fixate_plotting_max=fixate_plotting_max,
    )


@app.command("one_cohort_statistics")
def one_cohort_statistics_command(
    cohort: str = typer.Option(..., "--cohort", help="Path to the cohort directory."),
    reference_mouse: str = typer.Option(..., "--reference_mouse", help="Path to the reference mouse directory."),
    result_dir: str = typer.Option(..., "--result_dir", help="Path to the result folder."),
    upper_limit: Optional[float] = typer.Option(None, "--upper_limit", help="Path to the result file."),
    save_statics: bool = typer.Option(False, "--save_statics", help="Save the statistics to a file."),
    mahalanobis_summary_csv: Optional[str] = typer.Option(None, "--mahalanobis_summary_csv", help="Optional summary CSV path for leave-one-out Mahalanobis outlier statistics."),
    mahal_threshold: float = typer.Option(5.0, "--mahal_threshold", help="Threshold used to summarize one-cohort Mahalanobis outlier burden."),
    skip_variability_gif: bool = typer.Option(False, "--skip_variability_gif", help="Skip the cohort variability GIF and only compute numeric outputs."),
    fast_outliers: bool = typer.Option(False, "--fast_outliers", help="Use faster approximate one-cohort outlier computation based on a single shared GPA."),
) -> None:
    del save_statics
    one_cohort_stats_cmd.run(
        cohort=cohort,
        reference_mouse=reference_mouse,
        result_dir=result_dir,
        upper_limit=upper_limit,
        mahalanobis_summary_csv=mahalanobis_summary_csv,
        mahal_threshold=mahal_threshold,
        skip_variability_gif=skip_variability_gif,
        fast_outliers=fast_outliers,
    )


def main() -> None:
    print(f"[INFO] mouse_skull_pipelines version={_get_app_version()}")
    app()


if __name__ == "__main__":
    main()
