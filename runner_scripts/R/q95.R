#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

rm(list = ls())

# =========================
# CONFIG (YOUR SETUP)
# =========================
year_dirs <- list(
  "2024" = "/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/2024",
  "2023" = "/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/2023",
  "2022" = "/home/varhai/PRIMUS/data/83_IMPC_DATA/Bioimaging/SCREENad.Morpho&BodyComp/2022"
)

out_base <- "/home/varhai/PRIMUS/data/83_REQUESTS/_Statistics/SCIENTIFIC_PROJECTS/Vendula_Novosadova/STA-2026-009 Skull analysis KO variabiity/Results/2026-03-13"
dir.create(out_base, recursive = TRUE, showWarnings = FALSE)

PROGRESS_EVERY <- 10
OUTLIER_AREA_MAX <- 60

autumn_blue <- c(
  "tomato", "#FFA500", "#9ACD32", "#1E90FF",
  "maroon", "darkorange3", "darkred", "#07407b"
)

fmt_sec <- function(x) sprintf("%.1fs", x)

# =========================
# READ ALL YEARS (POOLED)
# =========================
read_all_years_df <- function(year_dirs) {
  df_all   <- data.table()
  info_all <- data.table()

  t0 <- proc.time()[["elapsed"]]
  message("Starting pooled read across years: ", paste(names(year_dirs), collapse = ", "))

  for (yy in names(year_dirs)) {
    root_dir <- year_dirs[[yy]]
    message("\n==============================")
    message("YEAR: ", yy)
    message("ROOT: ", root_dir)
    message("==============================")

    if (!dir.exists(root_dir)) stop("Missing year dir: ", root_dir)

    dirs <- list.dirs(root_dir, recursive = FALSE, full.names = TRUE)
    message("Found ", length(dirs), " top-level dirs")

    if (length(dirs) == 0) next

    for (i in seq_along(dirs)) {
      Dir <- dirs[[i]]
      if (i %% PROGRESS_EVERY == 1 || i == length(dirs)) {
        message(sprintf("-- [%s] Dir %d/%d: %s", yy, i, length(dirs), basename(Dir)))
      }

      files_df <- c(
        file.path(Dir, "_mahalanobis_results/females/mahalanobis_per_face.csv"),
        file.path(Dir, "_mahalanobis_results/males/mahalanobis_per_face.csv")
      )

      files_info <- c(
        file.path(Dir, "_mahalanobis_results/females/mahalanobis_area_gt_5.csv"),
        file.path(Dir, "_mahalanobis_results/males/mahalanobis_area_gt_5.csv")
      )

      exclude_ids <- character()

      # ---- info + excluded IDs (area_percent_mahal_gt_5 > OUTLIER_AREA_MAX) ----
      for (f in files_info) {
        if (file.exists(f)) {
          tmp_info <- fread(f)
          if (!("subject" %in% names(tmp_info))) next

          tmp_info[, c("Gen", "Gender") := tstrsplit(subject, "_", fixed = TRUE, keep = 1:2)]
          tmp_info[, ID := fifelse(
            grepl("^[^_]+_[^_]+_.+$", subject),
            sub("^[^_]+_[^_]+_", "", subject),
            NA_character_
          )]
          exclude_ids <- c(exclude_ids, tmp_info[area_percent_mahal_gt_5 > OUTLIER_AREA_MAX, ID])

          tmp_info[, year := yy]
          info_all <- rbindlist(list(info_all, tmp_info), use.names = TRUE, fill = TRUE)
        }
      }

      exclude_ids <- unique(exclude_ids)

      # ---- per_face filtered early ----
      for (f in files_df) {
        if (file.exists(f)) {
          tmp_df <- fread(f)
          if (!("subject" %in% names(tmp_df))) next

          tmp_df[, ID := fifelse(
            grepl("^[^_]+_[^_]+_.+$", subject),
            sub("^[^_]+_[^_]+_", "", subject),
            NA_character_
          )]
          if (length(exclude_ids) > 0) tmp_df <- tmp_df[!ID %in% exclude_ids]
          tmp_df[, ID := NULL]

          tmp_df[, year := yy]
          df_all <- rbindlist(list(df_all, tmp_df), use.names = TRUE, fill = TRUE)
        }
      }
    }
  }

  message("\nFinished pooled read in ", fmt_sec(proc.time()[["elapsed"]] - t0))
  message("Total pooled df rows:   ", format(nrow(df_all), big.mark = ","))
  message("Total pooled info rows: ", format(nrow(info_all), big.mark = ","))

  list(df = df_all, info = info_all)
}

res <- read_all_years_df(year_dirs)
df <- res$df
setDT(df)

# =========================
# DERIVATIONS (AS IN q95.R)
# =========================

# Create group (same as original)
df[, group := fifelse(startsWith(subject, "wt_F_"), "wt_F",
                      fifelse(startsWith(subject, "wt_M_"), "wt_M",
                              fifelse(grepl("_F_", subject), "gene_F",
                                      fifelse(grepl("_M_", subject), "gene_M", NA_character_))))]

df <- df[!is.na(group)]

# ---- Summarise pooled (WT + gene) for plotting only ----
message("\nSummarising pooled df (for density plot)…")
t_sum <- proc.time()[["elapsed"]]
summ <- df[, .(
  mean   = mean(mahalanobis, na.rm = TRUE),
  median = median(mahalanobis, na.rm = TRUE),
  q75    = quantile(mahalanobis, 0.75, na.rm = TRUE),
  q95    = quantile(mahalanobis, 0.95, na.rm = TRUE),
  sd     = sd(mahalanobis, na.rm = TRUE)
), by = .(group, face_index)]
message("summ rows: ", nrow(summ), " (", fmt_sec(proc.time()[["elapsed"]] - t_sum), ")")

summ_long <- summ %>%
  pivot_longer(cols = c(mean, median, q75, q95, sd),
               names_to = "metric", values_to = "value") %>%
  mutate(sex = ifelse(grepl("_F$", group), "Female", "Male"))

# ---- Plot pooled distribution (ALL YEARS) ----
message("Saving pooled density plot…")
p <- ggplot(summ_long, aes(x = value, color = group)) +
  geom_density(linewidth = 1.1, adjust = 1.2) +
  geom_vline(
    data = summ_long %>%
      group_by(group, metric, sex) %>%
      summarise(mu = mean(value, na.rm = TRUE), .groups = "drop"),
    aes(xintercept = mu, color = group),
    linetype = "dashed",
    linewidth = 0.8,
    show.legend = FALSE
  ) +
  facet_grid(sex ~ metric, scales = "free_x") +
  scale_color_manual(values = autumn_blue) +
  theme_light() +
  labs(
    x = "Value",
    y = "Density",
    color = "Group",
    title = paste0("Distribution of face_index summary statistics (ALL YEARS: ",
                   paste(names(year_dirs), collapse = "+"), ")")
  ) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(out_base, paste0(Sys.Date(), "_ALL_YEARS_Distribution_face_index.jpg")),
  p, width = 15, height = 10, units = "cm"
)

# ---- WT q95 thresholds (pooled WT only) ----
setDT(summ)
summ[, c("type", "Gender") := tstrsplit(group, "_", fixed = TRUE)]
summ1_wt <- summ[type == "wt", .(Gender, face_index, q95)]

# ---- Gene animals vs pooled WT q95 ----
df[, Gender := tstrsplit(group, "_", fixed = TRUE)[[2]]]
df_gene <- df[startsWith(group, "gene_")]

setkey(summ1_wt, Gender, face_index)
setkey(df_gene, Gender, face_index)

df_gene <- summ1_wt[df_gene, on = .(Gender, face_index)]
df_gene[, mahal_gt_q95 := fifelse(mahalanobis > q95, mahalanobis, NA_real_)]

stats_gt_q95 <- df_gene[!is.na(mahal_gt_q95), .(
  n_gt_q95      = .N,
  mean_gt_q95   = mean(mahal_gt_q95, na.rm = TRUE),
  median_gt_q95 = median(mahal_gt_q95, na.rm = TRUE),
  q95_gt_q95    = quantile(mahalanobis, 0.95, na.rm = TRUE)
), by = .(Gender, face_index)]

stats_total <- df_gene[, .(
  n_total    = .N,
  mean_all   = mean(mahalanobis, na.rm = TRUE),
  median_all = median(mahalanobis, na.rm = TRUE),
  q95_all    = quantile(mahalanobis, 0.95, na.rm = TRUE)
), by = .(Gender, face_index)]

comparison <- merge(
  stats_total,
  stats_gt_q95,
  by = c("Gender", "face_index"),
  all.x = TRUE
)

comparison[, n_gt_q95_scaled :=
             1.5 + (n_gt_q95 - min(n_gt_q95, na.rm = TRUE)) /
             (max(n_gt_q95, na.rm = TRUE) - min(n_gt_q95, na.rm = TRUE)) * (5 - 0.1)
]

# ---- Histogram of points used per face_index (gt q95) ----
comparison[is.na(n_gt_q95), n_gt_q95 := 0]
message("Saving pooled face_index gt-q95 count histogram...")
hist_p <- ggplot(comparison, aes(x = n_gt_q95)) +
  geom_histogram(bins = 30, color = "white", fill = "#1E90FF") +
  theme_light() +
  labs(
    x = "Number of points > q95 per face_index",
    y = "Count of face_index",
    title = paste0("Distribution of points above q95 per face_index (ALL YEARS: ",
                   paste(names(year_dirs), collapse = "+"), ")")
  ) +
  theme(
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(out_base, paste0(Sys.Date(), "_ALL_YEARS_face_index_gtq95count_hist.jpg")),
  hist_p, width = 14, height = 10, units = "cm"
)

# =========================
# EXPORT (POOLED OUTPUTS)
# =========================
out_dir <- file.path(out_base, paste0(Sys.Date(), "_ALL_YEARS_q95"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cols_to_export <- c("mean_gt_q95", "median_gt_q95", "n_gt_q95_scaled", "q95_gt_q95")

message("\nWriting pooled q95 outputs into: ", out_dir)
out_all <- data.table()
for (g in unique(comparison$Gender)) {
  for (col in cols_to_export) {
    subject_name <- paste0(g, "_", col, "_ALL_YEARS")

    out <- comparison[Gender == g, .(
      subject     = subject_name,
      face_index  = face_index,
      mahalanobis = get(col)
    )]

    out_all <- rbindlist(list(out_all, out), use.names = TRUE, fill = TRUE)
  }
}

fwrite(out_all, file.path(out_dir, "q95_ALL_YEARS.csv"))

message("\nDone.")
