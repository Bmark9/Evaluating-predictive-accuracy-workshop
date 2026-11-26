# =============================================================================
# Exercise: tuning on the validation set -> over-optimistic accuracy
# Part 1 (instructor): data generation
# -----------------------------------------------------------------------------
# Produces three CSV files in data/:
#   knn_train.csv       n = 200   used to fit KNN
#   knn_validation.csv  n = 60    used to tune K (deliberately small/noisy)
#   knn_test.csv        n = 4000  stands in for "future data" (large, so its
#                                 accuracy is close to the true accuracy)
#
# Outcome y is binary and the classes are exactly balanced in every file,
# so accuracy is a sensible main metric (chance level = 0.50).
#
# Why these settings produce the intended lesson
#   - Small validation set: each validation accuracy has SE of about
#     sqrt(.7 * .3 / 60) = 0.06, so luck matters a lot when picking a winner.
#   - Many candidate K values (26 odd values, 1-51) with a fairly flat
#     accuracy curve: several K are about equally good, so the winner is
#     largely decided by noise.
#   - 3 pure-noise predictors: make the problem harder and the curve flatter,
#     as in real data where not every predictor matters.
#   - Large test set: test accuracy is a near-noise-free yardstick.
# In simulations with these settings (60 replications), the validation
# accuracy of the selected K exceeded its test accuracy by ~7 percentage
# points on average, versus ~1 point for a K fixed in advance (K = 15).
#
# The seed was chosen so that this single dataset shows the pattern clearly:
# validation and test accuracy agree on average across K, but the selected
# K's validation accuracy is clearly too high. Part 2 (optional section)
# shows the pattern is systematic, not a property of this seed.
# =============================================================================

# ---- Settings -----------------------------------------------------------------
seed      <- 278           # chosen for a clear illustration (see check below)
n_train   <- 200
n_val     <- 60
n_test    <- 4000
n_noise   <- 3           # number of irrelevant predictors
sharpness <- 2           # higher = cleaner class boundary (higher accuracy)
ks        <- seq(1, 51, by = 2)   # odd K only: no tied votes with 2 classes

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

set.seed(seed)
train      <- simulate_balanced(n_train, n_noise, sharpness)
validation <- simulate_balanced(n_val,   n_noise, sharpness)
test       <- simulate_balanced(n_test,  n_noise, sharpness)

dir.create(out_dir, showWarnings = FALSE)
write.csv(train,      file.path(out_dir, "knn_train.csv"),      row.names = FALSE)
write.csv(validation, file.path(out_dir, "knn_validation.csv"), row.names = FALSE)
write.csv(test,       file.path(out_dir, "knn_test.csv"),       row.names = FALSE)

# ---- Instructor check: does this dataset show what we want? ----------------------
library(class)
acc_for_k <- function(k, newdata) {
  pred <- knn(train[, -ncol(train)], newdata[, -ncol(newdata)],
              cl = factor(train$y), k = k)
  mean(pred == newdata$y)
}
val_acc  <- sapply(ks, acc_for_k, newdata = validation)
test_acc <- sapply(ks, acc_for_k, newdata = test)
best     <- which.max(val_acc)   # which.max takes the first (smallest) K on ties

cat("\nClass balance (share y = 1): train", mean(train$y),
    "| validation", mean(validation$y), "| test", mean(test$y), "\n")
cat(sprintf("Mean (validation - test) across all K: %+.3f  (SD %.3f)\n",
            mean(val_acc - test_acc), sd(val_acc - test_acc)))
cat(sprintf("Selected K = %d: validation %.3f, test %.3f, gap %+.3f\n",
            ks[best], val_acc[best], test_acc[best], val_acc[best] - test_acc[best]))
cat(sprintf("Best possible test accuracy (K = %d): %.3f\n",
            ks[which.max(test_acc)], max(test_acc)))
