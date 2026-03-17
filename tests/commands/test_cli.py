from typer.testing import CliRunner

from mouse_skull_pipelines import cli


runner = CliRunner()


def test_compare_cohorts_group_dispatch(monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli.compare_cohorts_cmd, "run", fake_run)

    result = runner.invoke(
        cli.app,
        [
            "compare_cohorts",
            "--cohort_1",
            "c1",
            "--cohort_2",
            "c2",
            "--reference_mouse",
            "ref",
            "--result_dir",
            "out",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["method"] == "group"
    assert calls[0]["cohort_1"] == "c1"
    assert calls[0]["area_threshold"] == 5.0
    assert calls[0]["recompute_area_percent_threshold"] == 40.0
    assert calls[0]["recompute_face_percent_threshold"] == 40.0


def test_compare_cohorts_per_subject_dispatch(monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli.compare_cohorts_cmd, "run", fake_run)

    result = runner.invoke(
        cli.app,
        [
            "compare_cohorts",
            "--cohort_1",
            "c1",
            "--cohort_2",
            "c2",
            "--reference_mouse",
            "ref",
            "--result_dir",
            "out",
            "--method",
            "per_subject",
            "--ref_distribution",
            "cohort_1",
            "--fixate_plotting_max",
            "3.5",
            "--area_threshold",
            "6.25",
            "--recompute_area_percent_threshold",
            "55",
            "--recompute_face_percent_threshold",
            "30",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["method"] == "per_subject"
    assert calls[0]["ref_distribution"] == "cohort_1"
    assert calls[0]["fixate_plotting_max"] == 3.5
    assert calls[0]["area_threshold"] == 6.25
    assert calls[0]["recompute_area_percent_threshold"] == 55.0
    assert calls[0]["recompute_face_percent_threshold"] == 30.0


def test_generate_full_body_accepts_underscore_alias(monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli.generate_full_body_cmd, "run", fake_run)

    result = runner.invoke(
        cli.app,
        [
            "generate_full_body",
            "--cohort_dir",
            "cohort",
            "--ref_image_dir",
            "ref",
            "--cores",
            "12",
            "--keep-going",
            "--config-override",
            "a=1",
            "--config-override",
            "b=2",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["cohort_dir"] == "cohort"
    assert calls[0]["ref_image_dir"] == "ref"
    assert calls[0]["config_override"] == ["a=1", "b=2"]
    assert calls[0]["cores"] == 12
    assert calls[0]["keep_going"] is True
