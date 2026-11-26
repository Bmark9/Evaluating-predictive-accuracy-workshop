# =============================================================================
# Exercise: how big should the hold-out set be? (random forest vs Lasso)
# Part 1 (instructor): data generation
# -----------------------------------------------------------------------------
# Produces two CSV files in data/:
#   holdout_data.csv  n = 500    the data participants have: split it into a
#                                training part and a hold-out part
#   external.csv      n = 10000  "the future": a large external dataset that
#                                reveals the true performance of a fitted model
#
# Lesson: every case you hold out makes the performance estimate more precise,
# but the model you evaluate worse. Flexible models pay more for lost
# training data, so the hold-out size can change which model looks best.
#
# Data-generating process (all predictors uniform on [-1, 1]):
#   y = 1.5 x1 + x2 - x3                                   linear part
#     + a * (sin(pi x4 x5) + 2 * (x6 > 0) * x4)            nonlinear part
#     + N(0, sigma^2) noise
#   x7 ... x20 are pure noise predictors.
# Why these settings produce the intended lesson
#   - Lasso captures the linear part only, but needs little data to do so
#     and ignores the noise predictors: its learning curve is flat.
#   - Random forest can also capture the nonlinear part, but needs more data,
#     especially with 14 noise predictors: its learning curve is steep.
#   - With these settings the two learning curves cross at roughly 300
#     training cases (a hold-out share of ~40%). Trained on all 500 cases, the random forest is better;
#     trained on much less (i.e. a large hold-out set), Lasso is better.
#
# The seed was chosen so that this single dataset shows the pattern clearly
# (see 03_teacher_full_curves.R for the check).
# =============================================================================

# ---- Settings -----------------------------------------------------------------
seed       <- 16      # chosen for a clear illustration
n_data     <- 500
n_external <- 10000
n_pred     <- 20      # predictors x1 ... x20 (x7 ... x20 are noise)
a          <- 1.5     # strength of the nonlinear part
sigma      <- 0.8     # noise SD

out_dir <- "data"

# ---- Data-generating process ----------------------------------------------------
simulate <- function(n) {
  X <- matrix(runif(n * n_pred, -1, 1), n, n_pred,
              dimnames = list(NULL, paste0("x", seq_len(n_pred))))
  y <- 1.5 * X[, 1] + X[, 2] - X[, 3] +
    a * (sin(pi * X[, 4] * X[, 5]) + 2 * (X[, 6] > 0) * X[, 4]) +
    rnorm(n, 0, sigma)
  d <- data.frame(round(X, 3), y = round(y, 3))
  d
}

set.seed(seed)
holdout_data <- simulate(n_data)
external     <- simulate(n_external)

dir.create(out_dir, showWarnings = FALSE)
write.csv(holdout_data, file.path(out_dir, "holdout_data.csv"), row.names = FALSE)
write.csv(external,     file.path(out_dir, "external.csv"),     row.names = FALSE)
