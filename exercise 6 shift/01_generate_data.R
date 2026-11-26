# =============================================================================
# Exercise: distribution shift and external validation
# Part 1 (instructor): data generation
# -----------------------------------------------------------------------------
# Produces two CSV files in data/:
#   internal.csv  n = 1000  the development data: tune, internally validate,
#                           and finally refit both models on all of it
#   external.csv  n = 1000  data from a new setting ("another hospital"),
#                           used once, for external validation
#
# Variables (all predictors uniform on [-2, 2] unless stated otherwise):
#   x1, x2, x3  causes of y: y = 2 sin(x1) + x2^2 / 2 + x1 x3 / 2 + N(0, 1)
#   x4, x5      unrelated to y
#   m           a CONSEQUENCE of y, e.g. a biomarker that rises with the
#               condition. Internally:  m = y + N(0, 1)
#               Externally the marker is measured with a different, less
#               sensitive assay:        m = 0.4 y + 1 + N(0, 1)
#
# Why this produces the intended lesson
#   - The mechanism from the causes to y is identical in both settings, and so
#     is the distribution of x1 ... x5: a model built on the causes keeps its
#     performance externally.
#   - Internally, m is an excellent predictor (it tracks y closely), so the
#     model using all predictors clearly wins internal validation.
#   - Externally, the m-y relationship has changed. The all-predictors model
#     leans on m and its predictions are pulled off: it loses external
#     validation, even though its internal validation was done properly.
#
# Seed: patterns are robust across seeds (see check); the seed is fixed for
# reproducibility.
# =============================================================================

# ---- Settings -----------------------------------------------------------------
seed       <- 1
n_internal <- 1000
n_external <- 1000
marker_external <- c(slope = 0.4, intercept = 1)   # m = slope * y + intercept + noise

out_dir <- "data"

# ---- Data-generating process ----------------------------------------------------
simulate <- function(n, external = FALSE) {
  X <- matrix(runif(n * 5, -2, 2), n, 5, dimnames = list(NULL, paste0("x", 1:5)))
  y <- 2 * sin(X[, "x1"]) + X[, "x2"]^2 / 2 + X[, "x1"] * X[, "x3"] / 2 + rnorm(n, 0, 1)
  m <- if (external) {
    marker_external[["slope"]] * y + marker_external[["intercept"]] + rnorm(n, 0, 1)
  } else {
    y + rnorm(n, 0, 1)
  }
  data.frame(round(X, 3), m = round(m, 3), y = round(y, 3))
}

set.seed(seed)
internal <- simulate(n_internal)
external <- simulate(n_external, external = TRUE)

dir.create(out_dir, showWarnings = FALSE)
write.csv(internal, file.path(out_dir, "internal.csv"), row.names = FALSE)
write.csv(external, file.path(out_dir, "external.csv"), row.names = FALSE)

# ---- Instructor check -----------------------------------------------------------------
# Default random forests, trained on a random 80% of internal data and evaluated
# on the remaining 20% (internal) and on the external data.
library(ranger)
idx <- sample(n_internal, 0.8 * n_internal)
mse <- function(fit, d) mean((d$y - predict(fit, d)$predictions)^2)
fit_all    <- ranger(y ~ ., data = internal[idx, ])
fit_causal <- ranger(y ~ x1 + x2 + x3, data = internal[idx, ])
cat(sprintf("\nInternal MSE: all predictors %.3f | causes only %.3f\n",
            mse(fit_all, internal[-idx, ]), mse(fit_causal, internal[-idx, ])))
cat(sprintf("External MSE: all predictors %.3f | causes only %.3f\n",
            mse(fit_all, external), mse(fit_causal, external)))
