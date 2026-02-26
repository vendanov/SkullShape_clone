#!/usr/bin/env python3
"""
Summarize KO mice by mahalanobis area thresholds and year coverage.

Inputs (choose one of):
  - area CSV (e.g., mahalanobis_area_gt_5.csv) with columns:
      subject, area_percent_mahal_gt_5
  - cohorts root directory (like submit_compare_cohorts_batch.sh),
    optionally filtered by year in cohort folder name or year subfolder.

Outputs (CSV):
  - year_<YEAR>_summary.csv
  - year_<YEAR>_computed_subjects.csv
  - year_<YEAR>_missing_subjects.csv
  - ko_area_gt_<THRESH>.csv for each threshold
  - ko_gene_abnormal_ratios.csv
  - ko_threshold_comparison.csv
  - ko_threshold_counts.csv
  - ko_threshold_pairwise_overlap.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_THRESHOLDS = [0.25, 0.5, 1.0, 2.0]


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_area_csv(area_csv: Path, subject_col: str, area_col: str) -> Dict[str, float]:
    area_by_subject: Dict[str, float] = {}
    with area_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if subject_col not in reader.fieldnames or area_col not in reader.fieldnames:
            raise ValueError(
                f"CSV missing required columns. Found={reader.fieldnames}, "
                f"required={subject_col!r}, {area_col!r}"
            )
        for row in reader:
            subject = (row.get(subject_col) or "").strip()
            if not subject:
                continue
            area_val = _safe_float(row.get(area_col, ""))
            if area_val is None:
                continue
            area_by_subject[subject] = area_val
    return area_by_subject


def _collect_expected_subjects_for_cohort(cohort_dir: Path) -> List[str]:
    """Expected subjects are immediate subfolders in the cohort root (not derived from computations)."""
    ignore = {
        "_mahalanobis_results",
        "mouse_skull_workdir",
        "full_body_workdir",
        "males",
        "females",
        "Morphology",
        "BMD_TMD"
    }
    subjects: List[str] = []
    for p in cohort_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name in ignore or name.startswith("_") or name.endswith("+"):
            continue
        subjects.append(name)
    return sorted(set(subjects))


def _collect_cohort_dirs(cohorts_root: Path) -> Tuple[List[Path], bool]:
    direct_dirs = [p for p in cohorts_root.iterdir() if p.is_dir()]
    year_dirs = [p for p in direct_dirs if re.fullmatch(r"(19|20)\d{2}", p.name)]
    cohort_dirs: List[Path] = []
    if year_dirs:
        for year_dir in sorted(year_dirs):
            cohort_dirs.extend(p for p in year_dir.iterdir() if p.is_dir())
        cohort_dirs.extend(p for p in direct_dirs if p not in year_dirs)
    else:
        cohort_dirs = direct_dirs
    return sorted(set(cohort_dirs)), bool(year_dirs)


def _matches_year_filter(cohort_dir: Path, year_filter: Optional[str], year_dirs_present: bool) -> bool:
    if not year_filter:
        return True
    if year_filter in cohort_dir.name:
        return True
    if year_dirs_present:
        parent = cohort_dir.parent
        if re.fullmatch(r"(19|20)\d{2}", parent.name) and year_filter in parent.name:
            return True
    return False


def _collect_from_cohorts_root(
    cohorts_root: Path,
    year_filter: Optional[str],
    results_subdir: str,
    area_csv_name: str,
    subject_col: str,
    area_col: str,
) -> Tuple[Dict[str, float], List[str]]:
    area_by_subject: Dict[str, float] = {}
    expected_subjects: List[str] = []
    area_csvs: List[Path] = []

    cohort_dirs, year_dirs_present = _collect_cohort_dirs(cohorts_root)
    for cohort_dir in cohort_dirs:
        if not _matches_year_filter(cohort_dir, year_filter, year_dirs_present):
            continue
        expected_subjects.extend(_collect_expected_subjects_for_cohort(cohort_dir))
        results_dir = cohort_dir / results_subdir
        if results_dir.exists():
            direct = results_dir / area_csv_name
            if direct.exists():
                area_csvs.append(direct)
            else:
                area_csvs.extend(results_dir.rglob(area_csv_name))

    for area_csv in area_csvs:
        merged = _read_area_csv(area_csv, subject_col, area_col)
        area_by_subject.update(merged)

    return area_by_subject, sorted(set(expected_subjects))


def _read_expected_subjects(
    cohort_dir: Optional[Path],
    expected_subjects_file: Optional[Path],
    subject_col: str,
) -> Optional[List[str]]:
    if cohort_dir is not None:
        if not cohort_dir.exists():
            raise FileNotFoundError(f"Cohort dir not found: {cohort_dir}")
        subjects = [p.name for p in cohort_dir.iterdir() if p.is_dir()]
        return sorted(subjects)
    if expected_subjects_file is None:
        return None
    if not expected_subjects_file.exists():
        raise FileNotFoundError(f"Expected subjects file not found: {expected_subjects_file}")
    if expected_subjects_file.suffix.lower() in {".csv", ".tsv"}:
        with expected_subjects_file.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if subject_col not in reader.fieldnames:
                raise ValueError(
                    f"Expected subjects CSV missing column {subject_col!r}. "
                    f"Found={reader.fieldnames}"
                )
            return sorted({(row.get(subject_col) or "").strip() for row in reader if row.get(subject_col)})
    # plain text: one subject per line
    subjects = []
    with expected_subjects_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                subjects.append(line)
    return sorted(set(subjects))


def _year_from_subject(subject: str) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", subject)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _gender_from_subject(subject: str) -> str:
    tokens = re.split(r"[_\-]", subject)
    for token in tokens:
        upper = token.strip().upper()
        if upper in {"M", "MALE"}:
            return "M"
        if upper in {"F", "FEMALE"}:
            return "F"
    # substring fallback
    if re.search(r"\bMALE\b", subject, re.IGNORECASE):
        return "M"
    if re.search(r"\bFEMALE\b", subject, re.IGNORECASE):
        return "F"
    return "Unknown"


def _gene_from_subject(subject: str) -> str:
    return subject.split("_", 1)[0] if "_" in subject else subject


def _fmt_threshold(th: float) -> str:
    s = f"{th:g}"
    return s.replace(".", "_")


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pairwise_overlaps(sets_by_label: Dict[str, Set[str]]) -> List[Dict[str, object]]:
    labels = list(sets_by_label.keys())
    rows: List[Dict[str, object]] = []
    for i, a in enumerate(labels):
        for b in labels[i:]:
            inter = sets_by_label[a] & sets_by_label[b]
            rows.append({"threshold_a": a, "threshold_b": b, "intersection_count": len(inter)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize KO mice by mahalanobis area thresholds and year coverage."
    )
    parser.add_argument("--area-csv", type=Path, help="Path to mahalanobis_area_gt_5.csv")
    parser.add_argument("--cohorts-root", type=Path, help="Root directory used by submit_compare_cohorts_batch.sh")
    parser.add_argument(
        "--year-filter",
        type=str,
        default="",
        help="Filter cohorts by year in folder name (or year subfolder). Use 'all' to disable.",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Combine all year folders under cohorts root (disables year filtering).",
    )
    parser.add_argument("--results-subdir", default="_mahalanobis_results", help="Results folder under each cohort")
    parser.add_argument("--area-csv-name", default="mahalanobis_area_gt_5.csv", help="Area CSV filename")
    parser.add_argument("--cohort-dir", type=Path, help="Cohort directory (expected subjects = folder names)")
    parser.add_argument("--expected-subjects", type=Path, help="Text/CSV of expected subjects")
    parser.add_argument("--subject-column", default="subject", help="Column name for subject id")
    parser.add_argument("--area-column", default="area_percent_mahal_gt_5", help="Column name for area percent")
    parser.add_argument("--year", type=int, default=2024, help="Year to summarize")
    parser.add_argument(
        "--year-source",
        choices=["subject", "cohort"],
        default="cohort",
        help="Where to derive year for expected/computed counts",
    )
    parser.add_argument(
        "--thresholds",
        default="0.25,0.5,1,2",
        help="Comma-separated thresholds for area_percent_mahal_gt_5",
    )
    parser.add_argument(
        "--ko-gene-exclude",
        default="WT",
        help="Comma-separated gene names to exclude from KO lists (case-insensitive)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory for CSVs")
    args = parser.parse_args()

    if not args.area_csv and not args.cohorts_root:
        raise SystemExit("ERROR: Provide --area-csv or --cohorts-root.")

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    thresholds = sorted(set(thresholds))
    exclude_genes = {g.strip().upper() for g in args.ko_gene_exclude.split(",") if g.strip()}

    area_by_subject: Dict[str, float] = {}
    expected_subjects: Optional[List[str]] = None
    year_filter_input = (args.year_filter or "").strip()
    all_years_requested = args.all_years or year_filter_input.lower() in {"all", "any", "*", "none"}
    if all_years_requested:
        year_filter = ""
    else:
        year_filter = year_filter_input or str(args.year)
    all_years_mode = not year_filter
    if args.cohorts_root:
        cohorts_root = args.cohorts_root.expanduser().resolve()
        if not cohorts_root.exists():
            raise FileNotFoundError(f"Cohorts root not found: {cohorts_root}")
        area_by_subject, expected_subjects = _collect_from_cohorts_root(
            cohorts_root,
            year_filter if year_filter else None,
            args.results_subdir,
            args.area_csv_name,
            args.subject_column,
            args.area_column,
        )
    if args.area_csv:
        area_csv = args.area_csv.expanduser().resolve()
        if not area_csv.exists():
            raise FileNotFoundError(f"Area CSV not found: {area_csv}")
        area_by_subject.update(_read_area_csv(area_csv, args.subject_column, args.area_column))

    if expected_subjects is None:
        expected_subjects = _read_expected_subjects(
            args.cohort_dir.expanduser().resolve() if args.cohort_dir else None,
            args.expected_subjects.expanduser().resolve() if args.expected_subjects else None,
            args.subject_column,
        )

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Year summary
    summary_label = "ALL" if all_years_mode and args.year_source == "cohort" else str(args.year)
    if expected_subjects is not None:
        if args.year_source == "cohort":
            expected_yr = list(expected_subjects)
        else:
            expected_yr = [s for s in expected_subjects if _year_from_subject(s) == args.year]
        computed_yr = [s for s in expected_yr if s in area_by_subject]
        missing_yr = [s for s in expected_yr if s not in area_by_subject]
    else:
        expected_yr = []
        if all_years_mode and args.year_source == "cohort":
            computed_yr = list(area_by_subject)
        else:
            computed_yr = [s for s in area_by_subject if _year_from_subject(s) == args.year]
        missing_yr = []

    _write_csv(
        out_dir / f"year_{summary_label}_summary.csv",
        [{
            "year": summary_label,
            "expected_count": len(expected_yr),
            "computed_count": len(computed_yr),
            "missing_count": len(missing_yr),
        }],
        ["year", "expected_count", "computed_count", "missing_count"],
    )
    _write_csv(
        out_dir / f"year_{summary_label}_computed_subjects.csv",
        [{"subject": s} for s in computed_yr],
        ["subject"],
    )
    _write_csv(
        out_dir / f"year_{summary_label}_missing_subjects.csv",
        [{"subject": s} for s in missing_yr],
        ["subject"],
    )

    # Per-gene abnormal ratios (KO only)
    if expected_subjects is not None and expected_yr:
        gene_total_subjects: Dict[str, List[str]] = {}
        for subject in expected_yr:
            gene = _gene_from_subject(subject)
            if gene.upper() in exclude_genes:
                continue
            gene_total_subjects.setdefault(gene, []).append(subject)
    else:
        # Fallback: compute ratios over computed subjects only.
        gene_total_subjects = {}
        for subject in area_by_subject.keys():
            gene = _gene_from_subject(subject)
            if gene.upper() in exclude_genes:
                continue
            gene_total_subjects.setdefault(gene, []).append(subject)

    gene_ratio_rows: List[Dict[str, object]] = []
    for gene, subjects in sorted(gene_total_subjects.items()):
        total = len(subjects)
        computed_subjects = [s for s in subjects if s in area_by_subject]
        computed_count = len(computed_subjects)
        row: Dict[str, object] = {
            "gene": gene,
            "total_subjects": total,
            "computed_subjects": computed_count,
        }
        for th in thresholds:
            abnormal = sum(
                1 for s in computed_subjects if area_by_subject.get(s, float("-inf")) > th
            )
            ratio_total = (abnormal / total) if total else 0.0
            row[f"abnormal_gt_{_fmt_threshold(th)}"] = abnormal
            row[f"ratio_gt_{_fmt_threshold(th)}"] = ratio_total
        gene_ratio_rows.append(row)

    _write_csv(
        out_dir / "ko_gene_abnormal_ratios.csv",
        gene_ratio_rows,
        ["gene", "total_subjects", "computed_subjects"]
        + [f"abnormal_gt_{_fmt_threshold(th)}" for th in thresholds]
        + [f"ratio_gt_{_fmt_threshold(th)}" for th in thresholds],
    )

    # KO threshold lists
    ko_rows: List[Dict[str, object]] = []
    for subject, area_val in area_by_subject.items():
        gene = _gene_from_subject(subject)
        if gene.upper() in exclude_genes:
            continue
        ko_rows.append(
            {
                "subject": subject,
                "gene": gene,
                "gender": _gender_from_subject(subject),
                "area_percent_mahal_gt_5": area_val,
            }
        )

    sets_by_threshold: Dict[str, Set[str]] = {}
    for th in thresholds:
        label = f"gt_{_fmt_threshold(th)}"
        matched = [r for r in ko_rows if float(r["area_percent_mahal_gt_5"]) > th]
        sets_by_threshold[label] = {r["subject"] for r in matched}
        _write_csv(
            out_dir / f"ko_area_gt_{_fmt_threshold(th)}.csv",
            matched,
            ["subject", "gene", "gender", "area_percent_mahal_gt_5"],
        )

    # Comparison table per subject
    comparison_rows = []
    for row in ko_rows:
        area_val = float(row["area_percent_mahal_gt_5"])
        flags = {f"gt_{_fmt_threshold(th)}": area_val > th for th in thresholds}
        comparison_rows.append({**row, **flags})
    _write_csv(
        out_dir / "ko_threshold_comparison.csv",
        comparison_rows,
        ["subject", "gene", "gender", "area_percent_mahal_gt_5"]
        + [f"gt_{_fmt_threshold(th)}" for th in thresholds],
    )

    # Counts and overlaps
    _write_csv(
        out_dir / "ko_threshold_counts.csv",
        [{"threshold": k, "count": len(v)} for k, v in sets_by_threshold.items()],
        ["threshold", "count"],
    )
    _write_csv(
        out_dir / "ko_threshold_pairwise_overlap.csv",
        _pairwise_overlaps(sets_by_threshold),
        ["threshold_a", "threshold_b", "intersection_count"],
    )

    # Console summary
    print(f"[Year {summary_label}] expected={len(expected_yr)} computed={len(computed_yr)} missing={len(missing_yr)}")
    if expected_subjects is None:
        print("[WARN] No cohort/expected subject list provided; 'missing' counts are not computed.")
    print("[KO threshold counts]")
    for k in sorted(sets_by_threshold.keys(), key=lambda x: float(x.split('_', 1)[1].replace('_', '.'))):
        print(f"  {k}: {len(sets_by_threshold[k])}")
    print(f"[OK] Wrote CSVs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
