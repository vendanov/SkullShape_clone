#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
})

rm(list = ls())

# =========================
# CONFIG (YOUR SETUP)
# =========================
year_dirs <- list(
  "2024" = "/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/2024",
  "2023" = "/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/2023"
)

out_base <- "/home/varhai/PRIMUS/data/83_REQUESTS/_Statistics/SCIENTIFIC_PROJECTS/Vendula_Novosadova/STA-2026-009 Skull analysis KO variabiity/Results/2026-02-24"
dir.create(out_base, recursive = TRUE, showWarnings = FALSE)

PROGRESS_EVERY <- 10
OUTLIER_AREA_MAX <- 60

# =========================
# HELPERS
# =========================
read_area1_one_year <- function(root_dir, year_label) {
  message("\n==============================")
  message("YEAR: ", year_label)
  message("ROOT: ", root_dir)
  message("==============================")

  if (!dir.exists(root_dir)) stop("Missing year dir: ", root_dir)

  dirs <- list.dirs(root_dir, recursive = FALSE, full.names = TRUE)
  message("Found ", length(dirs), " top-level dirs")

  df_year <- data.table()

  for (i in seq_along(dirs)) {
    Dir <- dirs[[i]]
    if (i %% PROGRESS_EVERY == 1 || i == length(dirs)) {
      message(sprintf("-- [%s] Dir %d/%d: %s", year_label, i, length(dirs), basename(Dir)))
    }

    files_df <- c(
      file.path(Dir, "_mahalanobis_results/females/mahalanobis_per_face.csv"),
      file.path(Dir, "_mahalanobis_results/males/mahalanobis_per_face.csv")
    )

    files_info <- c(
      file.path(Dir, "_mahalanobis_results/females/mahalanobis_area_gt_5.csv"),
      file.path(Dir, "_mahalanobis_results/males/mahalanobis_area_gt_5.csv")
    )

    bad_ids <- character()
    exclude_ids <- character()

    # collect IDs where (Gen != wt) AND area_percent_mahal_gt_5 > 1
    for (f in files_info) {
      if (file.exists(f)) {
        tmp_info <- fread(f)
        if (!("subject" %in% names(tmp_info))) next

        tmp_info[, c("Gen", "Gender", "ID") := tstrsplit(subject, "_", fixed = TRUE)]
        exclude_ids <- c(exclude_ids, tmp_info[area_percent_mahal_gt_5 > OUTLIER_AREA_MAX, ID])
        bad_ids <- c(bad_ids, tmp_info[Gen != "wt" & area_percent_mahal_gt_5 > 1, ID])
      }
    }
    bad_ids <- setdiff(unique(bad_ids), unique(exclude_ids))

    # keep only per_face rows for those bad_ids
    for (f in files_df) {
      if (file.exists(f)) {
        tmp_df <- fread(f)
        if (!("subject" %in% names(tmp_df))) next

        tmp_df[, ID := tstrsplit(subject, "_", fixed = TRUE)[[3]]]
        tmp_df <- tmp_df[ID %in% bad_ids]
        tmp_df[, ID := NULL]

        tmp_df[, year := year_label]
        df_year <- rbindlist(list(df_year, tmp_df), use.names = TRUE, fill = TRUE)
      }
    }
  }

  setDT(df_year)

  # Create group
  df_year[, group := fifelse(startsWith(subject, "wt_F_"), "wt_F",
                             fifelse(startsWith(subject, "wt_M_"), "wt_M",
                                     fifelse(grepl("_F_", subject), "gene_F",
                                             fifelse(grepl("_M_", subject), "gene_M", NA_character_))))]

  df_year <- df_year[!is.na(group)]
  df_year
}

make_area1_output <- function(df_in, label_for_filename) {
  summ <- df_in[, .(
    n      = .N,
    mean   = mean(mahalanobis, na.rm = TRUE),
    median = median(mahalanobis, na.rm = TRUE),
    q95    = quantile(mahalanobis, 0.95, na.rm = TRUE),
    q90    = quantile(mahalanobis, 0.90, na.rm = TRUE)
  ), by = .(group, face_index)]

  summ_long <- as.data.frame(summ) %>%
    dplyr::select(-n) %>%
    pivot_longer(cols = c(mean, median, q95, q90),
                 names_to = "metric",
                 values_to = "mahalanobis") %>%
    mutate(
      sex = ifelse(grepl("_F$", group), "F", "M"),
      subject = paste(sex, metric, "area1", sep = "_")
    ) %>%
    dplyr::select(subject, face_index, mahalanobis)

  out_file <- file.path(out_base, paste0("area1_", label_for_filename, ".csv"))
  fwrite(as.data.table(summ_long), out_file)
  message("Wrote: ", out_file)
}

# =========================
# RUN (ALL YEARS -> ONE FILE)
# =========================
df_all <- rbindlist(
  Map(read_area1_one_year, year_dirs, names(year_dirs)),
  use.names = TRUE, fill = TRUE
)

message("\nTotal pooled rows (all years): ", format(nrow(df_all), big.mark = ","))
message("Total pooled subjects (area1): ", length(unique(df_all$subject)))

# 1) Combined / pooled output across all years
make_area1_output(df_all, "ALL_YEARS_POOLED")

# 2) Optional: per-year outputs (sanity check; remove if you don’t want these)
for (yy in unique(df_all$year)) {
  make_area1_output(df_all[year == yy], paste0("YEAR_", yy))
}

message("\nDone.")
