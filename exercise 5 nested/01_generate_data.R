# =============================================================================
# Exercise: an honest test set after tuning on the validation set
# Part 1 (instructor): data generation
# -----------------------------------------------------------------------------
# Follow-up to the leakage exercise (tuning on the validation set ->
# over-optimistic accuracy). Same data, plus a separate hold-out test set.
#
# Produces four CSV files in data/:
#   knn_train.csv         n = 200   fit KNN                 (same as before)
#   knn_validation.csv    n = 60    tune K                  (same as before)
#   knn_holdout_test.csv  n = 60    NEW: untouched until K is chosen; same
#                                   size as the validation set
#   knn_population.csv    n = 4000  stands in for the true accuracy (same data
#                                   as knn_test.csv in the previous exercise)
#
# Outcome y is binary and the classes are exactly balanced in every file.
#
# Why this produces the intended lesson
#   - K is selected on the validation set, so the validation accuracy of the
#     selected K is optimistic (previous exercise).
#   - The hold-out test set played no part in choosing K, so its accuracy is
#     an unbiased estimate of the true accuracy. With 60 cases it is still
#     noisy (95% CI about +/- 11 points), but it is right on average.
#   - The test set has the same size as the validation set: the difference
#     between the two estimates is not precision but selection.
#
# Seeds: `seed` reproduces the previous exercise's data exactly (same draws
# in the same order). `test_seed` was chosen so that this single test set
# lands close to the true accuracy (73.3% vs 73.5% for the selected K = 7).
# The optional part of the exercise shows that test estimates are right on
# average, not just for this seed.
# =============================================================================

# ---- Settings -----------------------------------------------------------------
seed      <- 278         # same as the previous exercise
test_seed <- 8           # chosen for a clear illustration (see check below)
n_train   <- 200
n_val     <- 60
n_pop     <- 4000
n_test    <- n_val       # hold-out test set: same size as the validation set
n_noise   <- 3           # number of irrelevant predictors
sharpness <- 2           # higher = cleaner class boundary (higher accuracy)
ks        <- 1:30        # K values considered in the exercise

out_dir <- "data"

# ---- Data-generating process ----------------------------------------------------
# Two informative predictors (x1, x2) with a curved class boundary, plus noise
# predictors x3..x5. All predictors are on the same scale (-2 to 2), so KNN
# distances need no standardisation.
#   P(y = 1 | x) = logistic(sharpness * (1.5 * sin(1.5 * x1) + x2^2 - 1.3))
simulate_balanced <- function(n, n_noise, sharpness) {
  stopifnot(n %% 2 == 0)
  pool <- NULL
  repeat {
    m  <- 4 * n
    x1 <- runif(m, -2, 2)
    x2 <- runif(m, -2, 2)
    p  <- plogis(sharpness * (1.5 * sin(1.5 * x1) + x2^2 - 1.3))
    y  <- rbinom(m, 1, p)
    noise <- matrix(runif(m * n_noise, -2, 2), m, n_noise)
    d <- data.frame(x1, x2, noise, y)
    names(d) <- c(paste0("x", seq_len(2 + n_noise)), "y")
    pool <- rbind(pool, d)
    if (sum(pool$y == 0) >= n / 2 && sum(pool$y == 1) >= n / 2) break
  }
  # Exactly n/2 per class, rows shuffled
  d <- rbind(pool[pool$y == 0, ][seq_len(n / 2), ],
             pool[pool$y == 1, ][seq_len(n / 2), ])
  d <- d[sample(nrow(d)), ]
  d[, -ncol(d)] <- round(d[, -ncol(d)], 3)   # tidier CSVs
  rownames(d) <- NULL
  d
}

# Same draws, in the same order, as the previous exercise
set.seed(seed)
train      <- simulate_balanced(n_train, n_noise, sharpness)
validation <- simulate_balanced(n_val,   n_noise, sharpness)
population <- simulate_balanced(n_pop,   n_noise, sharpness)

# New: the hold-out test set
set.seed(test_seed)
holdout_test <- simulate_balanced(n_test, n_noise, sharpness)

dir.create(out_dir, showWarnings = FALSE)
write.csv(train,        file.path(out_dir, "knn_train.csv"),        row.names = FALSE)
write.csv(validation,   file.path(out_dir, "knn_validation.csv"),   row.names = FALSE)
write.csv(holdout_test, file.path(out_dir, "knn_holdout_test.csv"), row.names = FALSE)
write.csv(population,   file.path(out_dir, "knn_population.csv"),   row.names = FALSE)

# ---- Instructor check: does this dataset show what we want? ----------------------
library(class)
predictors <- setdiff(names(train), "y")
acc_for_k <- function(k, newdata) {
  pred <- knn(train[, predictors], newdata[, predictors], cl = factor(train$y), k = k)
  mean(pred == newdata$y)
}
set.seed(1)   # knn() breaks tied votes (even K) at random
val_acc <- sapply(ks, acc_for_k, newdata = validation)
best_k  <- ks[which.max(val_acc)]

ci95 <- function(acc, n) acc + c(-1, 1) * 1.96 * sqrt(acc * (1 - acc) / n)
test_acc <- acc_for_k(best_k, holdout_test)
true_acc <- acc_for_k(best_k, population)

cat("\nClass balance (share y = 1): train", mean(train$y), "| validation", mean(validation$y),
    "| test", mean(holdout_test$y), "| population", mean(population$y), "\n")
cat(sprintf("Selected K = %d\n", best_k))
cat(sprintf("  Validation accuracy %.3f  95%% CI [%.3f, %.3f]\n", max(val_acc), ci95(max(val_acc), n_val)[1], ci95(max(val_acc), n_val)[2]))
cat(sprintf("  Test accuracy       %.3f  95%% CI [%.3f, %.3f]\n", test_acc, ci95(test_acc, n_test)[1], ci95(test_acc, n_test)[2]))
cat(sprintf("  True accuracy       %.3f\n", true_acc))
