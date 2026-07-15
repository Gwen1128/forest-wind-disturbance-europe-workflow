# =========================
# bfast_global_fast_checkpoint_FIXED_DOMAIN.R
# Global sampling (no tiles) + storm_id=EventDate
# PU sampling: distance strata + month-weighted dates + anti-leakage + bin1 cap
# For each NEG: store ref_event_uid/ref_event_date/ref_storm_id/ref_event_dist_km
#
# FIX (DOMAIN BUG):
#   - template_m / forest_mask_m are now built from UNION COVER of main+supp (first layer),
#     instead of only r_main[[1]]. This prevents north/south being dropped simply because
#     the first main layer is NA in those regions.
#
# Notes:
#   - This script keeps your original "common_names = intersect(main, supp)" time alignment,
#     but prints a warning if any date layers are missing in either stack.
# =========================

suppressPackageStartupMessages({
  library(terra)
  library(lubridate)
})

logi <- function(...) { cat(...); cat("\n"); flush.console() }
set.seed(20251017)

# ---- terra IO behavior ----
terraOptions(todisk = TRUE, memfrac = 0.5)

# =========================================================
# paths (SET THESE)
# =========================================================
lai_main_dir <- Sys.getenv("DP_LAI_MAIN_DIR", unset="/projappl/project_2011073/LAI_Export")
lai_supp_dir <- Sys.getenv("DP_LAI_SUPP_DIR", unset="/projappl/project_2011073/LAI_EU_Supple")
forwind_shp  <- Sys.getenv("DP_FORWIND_SHP", unset="/projappl/project_2011073/FORWIND appendix/FORWIND_v2.shp")

output_dir <- Sys.getenv("DP_BFAST_OUTPUT_DIR", unset="bfast_output_global")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
output_file <- file.path(output_dir, "bfast_global.csv")
ckpt_file   <- file.path(output_dir, "checkpoint.txt")
logi(sprintf("🧾 output_file = %s", output_file))
logi(sprintf("🧾 ckpt_file   = %s", ckpt_file))

# =========================================================
# params
# =========================================================
TARGET_CRS <- "EPSG:3035"

neg_multiplier <- as.integer(Sys.getenv("DP_NEG_MULTIPLIER", unset="10"))

DIST_BINS_KM <- c(0, 1, 5, 20, Inf)
DIST_PROP    <- c(0.35, 0.30, 0.25, 0.10)
DIST_PROP    <- DIST_PROP / sum(DIST_PROP)

DATE_START <- as.Date("2003-01-01")
DATE_END   <- as.Date("2023-12-31")
DATE_STEP  <- "16 days"

BUFFER_DIST_KM <- 1
BUFFER_DAYS    <- 32

# harmonic regression params (match your old setup)
BFAST_ORDER <- 3
FREQ <- 23
MIN_NON_NA <- 30

# batching
BATCH_SIZE <- as.integer(Sys.getenv("DP_BFAST_BATCH_SIZE", unset="5000"))   # ✅ 建议 5000 起步；如果内存紧/IO抖动就降到 2000

# near-field cap
BIN1_CAP_MULT <- 1.5
BIN_TRANSFER_ORDER <- c("bin2","bin3","bin4")

# distance definition for ref_event_dist_km
# FALSE: centroid distance (fast). TRUE: polygon edge distance (slower).
USE_POLYGON_EDGE_DISTANCE <- FALSE

# =========================================================
# 0) resume logic
# =========================================================
START_ROW <- 1L
if (file.exists(output_file) && file.exists(ckpt_file)) {
  last_done <- suppressWarnings(as.integer(readLines(ckpt_file, warn = FALSE)[1]))
  if (is.finite(last_done) && last_done > 0) {
    START_ROW <- last_done + 1L
    logi(sprintf("🔁 Resume from START_ROW=%d (last_done=%d)", START_ROW, last_done))
  }
}

# =========================================================
# 1) list rasters (NO global resample/cover)
# =========================================================
logi("📦 Listing LAI files ...")
main_files <- sort(list.files(lai_main_dir, pattern = "ForestLAI_.*\\.tif$", full.names = TRUE))
supp_files <- sort(list.files(lai_supp_dir, pattern = "ForestLAI_.*\\.tif$", full.names = TRUE))
stopifnot(length(main_files) > 0, length(supp_files) > 0)

extract_date_name <- function(paths) {
  d <- sub(".*?(\\d{8}).*$", "\\1", basename(paths))
  paste0("d", d)
}

r_main <- rast(main_files)
r_supp <- rast(supp_files)
names(r_main) <- extract_date_name(main_files)
names(r_supp) <- extract_date_name(supp_files)

# --- Time alignment ---
n_main_all <- nlyr(r_main)
n_supp_all <- nlyr(r_supp)
logi(sprintf("🧾 main layers (raw) = %d | supp layers (raw) = %d", n_main_all, n_supp_all))

common_names <- intersect(names(r_main), names(r_supp))
if (length(common_names) == 0) stop("No common dates between main and supp.")
common_names <- common_names[order(match(common_names, names(r_main)))]

# Warn if missing dates in either stack (helps diagnose unexpected NA coverage)
miss_in_supp <- setdiff(names(r_main), names(r_supp))
miss_in_main <- setdiff(names(r_supp), names(r_main))
if (length(miss_in_supp) > 0) {
  logi(sprintf("⚠️ supp is missing %d date layers present in main. Example: %s",
               length(miss_in_supp), paste(head(miss_in_supp, 5), collapse=", ")))
}
if (length(miss_in_main) > 0) {
  logi(sprintf("⚠️ main is missing %d date layers present in supp. Example: %s",
               length(miss_in_main), paste(head(miss_in_main, 5), collapse=", ")))
}

r_main <- r_main[[common_names]]
r_supp <- r_supp[[common_names]]
logi(sprintf("✅ Common dates: %d layers", nlyr(r_main)))

# parse layer_dates from names like "dYYYYMMDD"
layer_dates <- as.Date(sub("^d", "", names(r_main)), format="%Y%m%d")
if (any(is.na(layer_dates))) stop("layer_dates parse failed: names must be dYYYYMMDD.")
N_T <- length(layer_dates)

logi("🧭 Global extent (native):"); print(ext(r_main[[1]])); flush.console()
logi("🧭 Global CRS (native):");   print(crs(r_main[[1]])); flush.console()

# =========================================================
# 2) FORWIND read + CRS match (native)
# =========================================================
logi("📥 Reading FORWIND ...")
fw <- vect(forwind_shp)
if (!("EventDate" %in% names(fw))) stop("FORWIND has no EventDate field.")

# robust date parse (handles "YYYY-MM-DD", "YYYY/MM/DD", etc)
fw$EventDate_dt <- suppressWarnings(lubridate::ymd(fw$EventDate))
if (all(is.na(fw$EventDate_dt))) stop("EventDate parse failed (ymd). Check shapefile.")
fw$EventDate_int <- as.numeric(fw$EventDate_dt)   # numeric days since 1970
fw$storm_id <- fw$EventDate_int

if (!terra::same.crs(fw, r_main[[1]])) fw <- project(fw, crs(r_main[[1]]))
logi(sprintf("✅ FORWIND polygons (global): %d", nrow(fw)))

# =========================================================
# 3) Metric template for sampling/distance ONLY
#    FIXED: union cover of (main[[1]], supp[[1]]) in TARGET_CRS
# =========================================================
logi(sprintf("🧭 Building metric template (%s) [FIX: union cover] ...", TARGET_CRS))

# 使用夏季图层（例如第12个图层，大约在6月底/7月初）来构建模板
# 这样可以防止高纬度地区因为冬季积雪/极夜导致的全部 NA 而被当成非森林丢弃
summer_idx <- min(12, nlyr(r_main)) 
m1 <- terra::project(r_main[[summer_idx]], TARGET_CRS, method="near")
s1 <- terra::project(r_supp[[summer_idx]], TARGET_CRS, method="near")

# make sure supp aligns to main grid
if (!all(res(m1) == res(s1))) {
  s1 <- terra::resample(s1, m1, method="near")
}

# union extent + extend to union
e_union <- terra::union(terra::ext(m1), terra::ext(s1))
m1u <- terra::extend(m1, e_union)
s1u <- terra::extend(s1, e_union)

# cover: wherever main is NA, take supp
template_m <- terra::cover(m1u, s1u)

rm(m1, s1, m1u, s1u); gc()

# --- sanity check: template domain in WGS84 ---
{
  e <- terra::ext(template_m)
  # ✅ FIXED: 转换为 data.frame 以完美适配 geom=c("x","y") 参数
  df_ext <- data.frame(
    x = c(e$xmin, e$xmin, e$xmax, e$xmax),
    y = c(e$ymin, e$ymax, e$ymin, e$ymax)
  )
  v <- terra::vect(df_ext, geom=c("x","y"), crs=terra::crs(template_m))
  xy <- terra::crds(terra::project(v, "EPSG:4326"))
  logi(sprintf("🧭 template_m lon range: %.2f .. %.2f", min(xy[,1]), max(xy[,1])))
  logi(sprintf("🧭 template_m lat range: %.2f .. %.2f", min(xy[,2]), max(xy[,2])))
  if (max(xy[,2]) < 70) {
    logi(sprintf("⚠️ template_m lat_max=%.2f < 70N. If you expect coverage to ~71N, your inputs are still truncated.", max(xy[,2])))
  }
}

fw_m <- fw
if (!terra::same.crs(fw_m, template_m)) fw_m <- terra::project(fw_m, crs(template_m))
fw_m$event_uid <- seq_len(nrow(fw_m))

forest_mask_m <- !is.na(values(template_m))

event_mask_m <- terra::rasterize(fw_m, template_m, field=1, background=0)
eventdate_m  <- terra::rasterize(fw_m, template_m, field="EventDate_int", background=NA, touches=TRUE)

pos_target <- event_mask_m
v <- values(pos_target); v[v != 1] <- NA; values(pos_target) <- v
dist_m <- terra::distance(pos_target)
rm(pos_target, v); gc()

# =========================================================
# 4) POS table (metric)
# =========================================================
pos_cells <- which(!is.na(values(eventdate_m)) & forest_mask_m)
logi(sprintf("🔎 pos_cells (metric global): %d", length(pos_cells)))
if (length(pos_cells) == 0) stop("No positive cells after rasterize.")

pos_dates_num <- values(eventdate_m)[pos_cells]             # numeric days since 1970
pos_event_time <- as.Date(pos_dates_num, origin="1970-01-01")
pos_xy_m <- xyFromCell(template_m, pos_cells)

pts_pos_df_m <- data.frame(
  x = pos_xy_m[,1],
  y = pos_xy_m[,2],
  label = 1L,
  event_time = as.Date(pos_event_time),
  storm_id = as.numeric(pos_dates_num),
  ref_event_uid = NA_integer_,
  ref_event_date = as.numeric(pos_dates_num),
  ref_storm_id = as.numeric(pos_dates_num),
  ref_event_dist_km = 0
)

n_neg <- nrow(pts_pos_df_m) * neg_multiplier
logi(sprintf("🎯 n_neg target: %d (ratio %d:1)", n_neg, neg_multiplier))

# =========================================================
# 5) NEG sampling pools by distance strata
# =========================================================
forest_cells <- which(forest_mask_m)
dist_km_all <- values(dist_m)[forest_cells] / 1000
dist_km_all[is.na(dist_km_all)] <- Inf

dist_bin <- cut(
  dist_km_all,
  breaks = DIST_BINS_KM,
  include.lowest = TRUE,
  right = TRUE,
  labels = paste0("bin", seq_len(length(DIST_BINS_KM) - 1))
)

bin_levels <- levels(dist_bin)
bin_pools <- lapply(bin_levels, function(bl) forest_cells[which(dist_bin == bl)])
names(bin_pools) <- bin_levels
bin_sizes <- sapply(bin_pools, length)
logi("📌 Forest cells per distance bin (global):"); print(bin_sizes); flush.console()

# allocate per bin + cap bin1
n_target_bin <- floor(n_neg * DIST_PROP)
residual <- n_neg - sum(n_target_bin)
if (residual > 0) {
  ord <- order(bin_sizes, decreasing=TRUE)
  for (k in seq_len(residual)) {
    j <- ord[(k - 1) %% length(ord) + 1]
    n_target_bin[j] <- n_target_bin[j] + 1L
  }
}

bin_idx <- setNames(seq_along(bin_levels), bin_levels)
if ("bin1" %in% bin_levels) {
  cap1 <- floor(bin_sizes[bin_idx[["bin1"]]] * BIN1_CAP_MULT)
  cap1 <- max(cap1, 0L)
  if (n_target_bin[bin_idx[["bin1"]]] > cap1) {
    overflow <- n_target_bin[bin_idx[["bin1"]]] - cap1
    n_target_bin[bin_idx[["bin1"]]] <- cap1
    for (bn in BIN_TRANSFER_ORDER) {
      if (overflow <= 0) break
      if (bn %in% bin_levels) {
        j <- bin_idx[[bn]]
        n_target_bin[j] <- n_target_bin[j] + overflow
        overflow <- 0
      }
    }
  }
}
stopifnot(sum(n_target_bin) == n_neg)
logi("🧪 n_target_bin after cap/transfer (global):")
print(setNames(n_target_bin, bin_levels)); flush.console()

# month-weighted dates from positive months
pos_months <- lubridate::month(pts_pos_df_m$event_time)
month_probs <- prop.table(table(pos_months))

all_dates <- seq(DATE_START, DATE_END, by=DATE_STEP)
cand_months <- lubridate::month(all_dates)
date_weights <- as.numeric(month_probs[as.character(cand_months)])
date_weights[is.na(date_weights)] <- 0
if (sum(date_weights) == 0) date_weights <- rep(1, length(all_dates))

sample_neg_pairs <- function(pool_cells, n_need, max_tries_mult=30L) {
  if (n_need <= 0) return(list(cells=integer(0), dates=as.Date(character(0))))
  if (length(pool_cells) == 0) return(list(cells=integer(0), dates=as.Date(character(0))))

  out_cells <- integer(0)
  out_dates <- as.Date(character(0))
  tries_left <- max_tries_mult * n_need
  pool_unique <- pool_cells

  while (length(out_cells) < n_need && tries_left > 0) {
    need_left <- n_need - length(out_cells)
    k <- min(max(2000L, need_left), 20000L)

    if (length(pool_unique) >= k) {
      cells_k <- sample(pool_unique, size=k, replace=FALSE)
      pool_unique <- setdiff(pool_unique, cells_k)
    } else {
      cells_k1 <- pool_unique
      pool_unique <- integer(0)
      k2 <- k - length(cells_k1)
      cells_k2 <- sample(pool_cells, size=k2, replace=TRUE)
      cells_k <- c(cells_k1, cells_k2)
    }

    dates_k <- sample(all_dates, size=length(cells_k), replace=TRUE, prob=date_weights)

    dist_km <- values(dist_m)[cells_k] / 1000
    dist_km[is.na(dist_km)] <- Inf

    ev_k <- values(eventdate_m)[cells_k]  # numeric days since 1970 or NA
    near_pos <- is.finite(dist_km) & (dist_km <= BUFFER_DIST_KM)

    ok <- ifelse(
      is.na(ev_k),
      TRUE,
      ifelse(
        near_pos,
        abs(as.numeric(dates_k) - ev_k) > BUFFER_DAYS,
        as.numeric(dates_k) != ev_k
      )
    )

    if (any(ok)) {
      out_cells <- c(out_cells, cells_k[ok])
      out_dates <- c(out_dates, dates_k[ok])
    }

    tries_left <- tries_left - length(cells_k)
  }

  if (length(out_cells) < n_need) {
    logi(sprintf("⚠️ Rejection sampling shortfall (need %d, got %d).", n_need, length(out_cells)))
  }

  out_cells <- out_cells[seq_len(min(n_need, length(out_cells)))]
  out_dates <- out_dates[seq_len(min(n_need, length(out_dates)))]
  list(cells=out_cells, dates=out_dates)
}

neg_cells <- integer(0)
neg_dates <- as.Date(character(0))
for (j in seq_along(bin_levels)) {
  res_j <- sample_neg_pairs(bin_pools[[j]], n_target_bin[j])
  neg_cells <- c(neg_cells, res_j$cells)
  neg_dates <- c(neg_dates, res_j$dates)
}
shortfall <- n_neg - length(neg_cells)
if (shortfall > 0) {
  logi(sprintf("ℹ️ Filling shortfall: %d", shortfall))
  res_fill <- sample_neg_pairs(forest_cells, shortfall)
  neg_cells <- c(neg_cells, res_fill$cells)
  neg_dates <- c(neg_dates, res_fill$dates)
}
if (length(neg_cells) > n_neg) {
  neg_cells <- neg_cells[seq_len(n_neg)]
  neg_dates <- neg_dates[seq_len(n_neg)]
}
stopifnot(length(neg_cells) == n_neg, length(neg_dates) == n_neg)

neg_xy_m <- xyFromCell(template_m, neg_cells)
pts_neg_df_m <- data.frame(
  x = neg_xy_m[,1],
  y = neg_xy_m[,2],
  label = 0L,
  event_time = as.Date(neg_dates),
  storm_id = as.numeric(as.Date(neg_dates))
)

# =========================================================
# 6) Assign ref_event_* for NEG
# =========================================================
logi("🔗 Assigning ref_event_* for NEG (global) ...")
neg_pts_v <- terra::vect(pts_neg_df_m, geom=c("x","y"), crs=crs(template_m))
fw_cent   <- terra::centroids(fw_m)

BLOCK_N <- 5000L
nP <- nrow(pts_neg_df_m)
idx_near <- integer(nP)
min_d_m  <- numeric(nP)

for (s in seq(1L, nP, by=BLOCK_N)) {
  e <- min(s + BLOCK_N - 1L, nP)
  dmat <- terra::distance(neg_pts_v[s:e,], fw_cent)
  jmin <- max.col(-dmat)
  idx_near[s:e] <- jmin
  min_d_m[s:e]  <- dmat[cbind(seq_len(nrow(dmat)), jmin)]
  rm(dmat); gc()
}
idx_near[idx_near < 1] <- 1L
idx_near[idx_near > nrow(fw_m)] <- nrow(fw_m)

pts_neg_df_m$ref_event_uid  <- as.integer(fw_m$event_uid[idx_near])
pts_neg_df_m$ref_event_date <- as.numeric(fw_m$EventDate_int[idx_near])
pts_neg_df_m$ref_storm_id   <- as.numeric(fw_m$storm_id[idx_near])

if (!USE_POLYGON_EDGE_DISTANCE) {
  pts_neg_df_m$ref_event_dist_km <- as.numeric(min_d_m) / 1000
} else {
  d_edge <- terra::distance(neg_pts_v, fw_m[idx_near,], pairwise=TRUE)
  pts_neg_df_m$ref_event_dist_km <- as.numeric(d_edge) / 1000
  rm(d_edge)
}
stopifnot(length(pts_neg_df_m$ref_event_dist_km) == n_neg)

rm(fw_cent, neg_pts_v, idx_near, min_d_m); gc()

# =========================================================
# 7) Combine POS+NEG (metric), project to native coords
# =========================================================
pts_df_m <- rbind(pts_pos_df_m, pts_neg_df_m)
pts_df_m$index <- seq_len(nrow(pts_df_m))

pts_df_m <- pts_df_m[, c(
  "index","x","y","label","event_time","storm_id",
  "ref_event_uid","ref_event_date","ref_storm_id","ref_event_dist_km"
)]
logi(sprintf("✅ Samples (global metric): pos=%d, neg=%d",
             sum(pts_df_m$label==1), sum(pts_df_m$label==0)))

pts_v_m <- terra::vect(pts_df_m, geom=c("x","y"), crs=crs(template_m))
pts_v_native <- terra::project(pts_v_m, crs(r_main[[1]]))
coords_native <- terra::crds(pts_v_native)

rm(pts_v_m, pts_v_native, template_m, event_mask_m, eventdate_m, dist_m, forest_mask_m); gc()

# =========================================================
# 8) FAST residual: precompute harmonic design matrix X
# =========================================================
logi("⚡ Precomputing harmonic design matrix ...")
t_idx <- 1:N_T
X <- matrix(1, nrow=N_T, ncol=1 + 2*BFAST_ORDER)
for (k in 1:BFAST_ORDER) {
  X[, 1 + k]               <- sin(2*pi*k*t_idx/FREQ)
  X[, 1 + BFAST_ORDER + k] <- cos(2*pi*k*t_idx/FREQ)
}
XtX_inv <- solve(crossprod(X))

# fast residual (fill NA by linear interp for speed)
fast_residual <- function(y) {
  y <- as.numeric(y)
  ok <- !is.na(y)
  if (sum(ok) < MIN_NON_NA) return(NULL)
  if (any(!ok)) {
    idx_ok <- which(ok)
    y[!ok] <- approx(x=idx_ok, y=y[idx_ok], xout=which(!ok), rule=2)$y
  }
  beta <- XtX_inv %*% crossprod(X, y)
  yhat <- X %*% beta
  as.numeric(y - yhat)
}

event_to_layer_idx <- function(event_time) {
  if (is.na(event_time)) return(NA_integer_)
  which.min(abs(as.integer(layer_dates - as.Date(event_time))))
}

# your spike clamp kept
spike_clamp_noqc <- function(v, pre_n=3, thr_abs=0.2, thr_rel=0.10, eps_back=0.05) {
  v <- as.numeric(v); n <- length(v)
  if (n < (pre_n + 2)) return(v)
  out <- v
  for (t in (pre_n + 1):(n - 1)) {
    pre <- v[(t - pre_n):(t - 1)]
    pre <- pre[!is.na(pre)]
    if (length(pre) == 0 || is.na(v[t]) || is.na(v[t + 1])) next
    baseline <- median(pre)
    thr <- max(thr_abs, thr_rel * max(baseline, 0))
    if ((v[t] < (baseline - thr)) && (v[t + 1] >= (baseline - eps_back))) {
      out[t] <- max(v[t], baseline - thr)
    }
  }
  out
}

custom_processing_fast <- function(y, event_time) {
  if (is.na(event_time)) return(list(break_date=NA, mag=NA, step="step3"))
  y <- spike_clamp_noqc(y)
  resid <- fast_residual(y)
  if (is.null(resid) || length(resid) != N_T) return(list(break_date=NA, mag=NA, step="step3"))
  j <- event_to_layer_idx(event_time)
  if (is.na(j)) return(list(break_date=NA, mag=NA, step="step3"))
  win <- (j-1):(j+1)
  win <- win[win >= 1 & win <= N_T]
  list(
    break_date = layer_dates[j],
    mag = mean(resid[win], na.rm=TRUE),
    step = "step2"
  )
}

# =========================================================
# 9) STREAMING extract main+supp per batch, cover in-memory, compute, write + checkpoint
# =========================================================

# write header if starting fresh
if (!file.exists(output_file) || START_ROW == 1L) {
  header <- data.frame(
    index=integer(0),
    x=numeric(0), y=numeric(0),
    label=integer(0),
    event_time=as.Date(character(0)),
    storm_id=numeric(0),
    ref_event_uid=integer(0),
    ref_event_date=numeric(0),
    ref_storm_id=numeric(0),
    ref_event_dist_km=numeric(0),
    break_date=as.Date(character(0)),
    mag=numeric(0),
    step_flag=character(0)
  )
  write.csv(header, output_file, row.names=FALSE)
  writeLines("0", ckpt_file)
}

total_rows <- nrow(pts_df_m)
logi(sprintf("🧮 Streaming extract+compute: total_rows=%d, START_ROW=%d, BATCH_SIZE=%d",
             total_rows, START_ROW, BATCH_SIZE))

for (i_start in seq(START_ROW, total_rows, by=BATCH_SIZE)) {
  i_end <- min(i_start + BATCH_SIZE - 1L, total_rows)
  logi(sprintf("📦 Batch: %d-%d / %d", i_start, i_end, total_rows))

  coords_batch <- coords_native[i_start:i_end, , drop=FALSE]
  meta_batch   <- pts_df_m[i_start:i_end, , drop=FALSE]

  # extract main + supp separately (NO resample/cover stack)
  ex_main <- terra::extract(r_main, coords_batch)
  ex_supp <- terra::extract(r_supp, coords_batch)
  if ("ID" %in% names(ex_main)) ex_main$ID <- NULL
  if ("ID" %in% names(ex_supp)) ex_supp$ID <- NULL

  mat_main <- as.matrix(ex_main)
  mat_supp <- as.matrix(ex_supp)

  # in-memory cover: use supp where main NA
  mat <- mat_main
  na_m <- is.na(mat)
  if (any(na_m)) mat[na_m] <- mat_supp[na_m]

  rm(ex_main, ex_supp, mat_main, mat_supp, na_m); gc()

  out_batch <- vector("list", length = nrow(meta_batch))
  for (ii in seq_len(nrow(meta_batch))) {
    row_info <- meta_batch[ii, ]
    y <- as.numeric(mat[ii, ])

    res <- tryCatch(
      custom_processing_fast(y, row_info$event_time),
      error=function(e) list(break_date=NA, mag=NA, step="step3")
    )

    out_batch[[ii]] <- data.frame(
      index=row_info$index,
      x=row_info$x, y=row_info$y,
      label=row_info$label,
      event_time=as.Date(row_info$event_time),
      storm_id=as.numeric(row_info$storm_id),
      ref_event_uid=as.integer(row_info$ref_event_uid),
      ref_event_date=as.numeric(row_info$ref_event_date),
      ref_storm_id=as.numeric(row_info$ref_storm_id),
      ref_event_dist_km=as.numeric(row_info$ref_event_dist_km),
      break_date=as.Date(res$break_date),
      mag=as.numeric(res$mag),
      step_flag=as.character(res$step)
    )
  }

  df_batch <- do.call(rbind, out_batch)
  write.table(df_batch, output_file, sep=",", row.names=FALSE, col.names=FALSE, append=TRUE)

  # checkpoint
  writeLines(as.character(i_end), ckpt_file)

  rm(df_batch, out_batch, mat, coords_batch, meta_batch); gc()
}

logi(sprintf("✅ Done: %s", output_file))