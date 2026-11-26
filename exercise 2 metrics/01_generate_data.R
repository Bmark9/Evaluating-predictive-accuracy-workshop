# =============================================================================
# Exercise: what you optimise is what you get (MSE vs MAE, accuracy vs Brier)
# Part 1 (instructor): data generation
# -----------------------------------------------------------------------------
# Produces four CSV files in data/:
#   reg_train.csv  n = 300    regression, predictor x, outcome y
#   reg_test.csv   n = 5000
#   clf_train.csv  n = 600    classification, predictors x1, x2, outcome y (0/1)
#   clf_test.csv   n = 5000   classes exactly balanced in both files
#
# Models participants compare (no tuning; K is given):
#   Regression:     KNN with K = 25, predicting the neighbours' MEAN vs MEDIAN
#   Classification: KNN with K = 1 vs K = 15, probability = share of
#                   neighbours with y = 1, class = probability > 0.5
#
# Why these settings produce the intended lesson
#   Regression: the noise is log-normal (right-skewed), so for every x the
#     mean of y lies well above the median. The neighbour mean targets the
#     conditional mean (minimises squared error); the neighbour median
#     targets the conditional median (minimises absolute error). With
#     symmetric noise both would coincide and the lesson would disappear.
#   Classification: the predictor space has two halves.
#     x1 < 0: a sharp, wiggly class boundary (true probabilities near 0 or 1).
#             Small K follows it best, so K = 1 wins on accuracy.
#     x1 > 0: the predictors carry no information (true probability 0.5).
#             Accuracy is 50% there whatever the model does, but Brier
#             punishes confident probabilities: K = 1 predicts only 0 or 1,
#             so K = 15 wins on Brier.
# Typical results across 200 simulated datasets (medians): the mean has ~4%
# lower MSE than the median, the median ~12% lower MAE than the mean; K = 1
# has ~4 points higher accuracy than K = 15, but a Brier score ~0.10 worse.
#
# The seed was chosen so that this single dataset shows both patterns clearly
# (see the check at the bottom).
# =============================================================================

# ---- Settings -----------------------------------------------------------------
seed <- 46   # chosen for a clear illustration (see check below)

# Regression
n_reg_train <- 300
n_reg_test  <- 5000
sdlog       <- 1          # skewness of the noise (log-normal)
f_reg       <- function(x) 2 * sin(x) + 0.5 * x
k_reg       <- 25

# Classification
n_clf_train <- 600
n_clf_test  <- 5000
p_clf <- function(x1, x2) {
  ifelse(x1 < 0,
         plogis(10 * sin(6 * x2 + 4 * x1)),   # sharp, wiggly boundary
         0.5)                                 # uninformative region
}
k_small <- 1
k_large <- 15

out_dir <- "data"

# ---- Data-generating functions ---------------------------------------------------
simulate_reg <- function(n) {
  x <- runif(n, 0, 10)
  y <- f_reg(x) + rlnorm(n, meanlog = 0, sdlog = sdlog)
  data.frame(x = round(x, 3), y = round(y, 3))
}

# Draws until both classes have n/2 cases, then keeps exactly n/2 of each
simulate_clf <- function(n) {
  stopifnot(n %% 2 == 0)
  pool <- NULL
  repeat {
    m  <- 4 * n
    x1 <- runif(m, -2, 2)
    x2 <- runif(m, -2, 2)
    y  <- rbinom(m, 1, p_clf(x1, x2))
    pool <- rbind(pool, data.frame(x1 = round(x1, 3), x2 = round(x2, 3), y))
    if (sum(pool$y == 0) >= n / 2 && sum(pool$y == 1) >= n / 2) break
  }
  d <- rbind(pool[pool$y == 0, ][seq_len(n / 2), ],
             pool[pool$y == 1, ][seq_len(n / 2), ])
  d <- d[sample(nrow(d)), ]
  rownames(d) <- NULL
  d
}

set.seed(seed)
reg_train <- simulate_reg(n_reg_train)
reg_test  <- simulate_reg(n_reg_test)
clf_train <- simulate_clf(n_clf_train)
clf_test  <- simulate_clf(n_clf_test)

dir.create(out_dir, showWarnings = FALSE)
write.csv(reg_train, file.path(out_dir, "reg_train.csv"), row.names = FALSE)
write.csv(reg_test,  file.path(out_dir, "reg_test.csv"),  row.names = FALSE)
write.csv(clf_train, file.path(out_dir, "clf_train.csv"), row.names = FALSE)
write.csv(clf_test,  file.path(out_dir, "clf_test.csv"),  row.names = FALSE)

# ---- Instructor check: does this dataset show what we want? ----------------------
# Same helper as in the solution script
knn_predict <- function(train_x, train_y, new_x, k, aggregate = mean) {
  train_x <- as.matrix(train_x)
  new_x   <- as.matrix(new_x)
  # squared Euclidean distances: rows = new cases, columns = training cases
  d2 <- outer(rowSums(new_x^2), rowSums(train_x^2), "+") - 2 * new_x %*% t(train_x)
  apply(d2, 1, function(d) aggregate(train_y[order(d)[seq_len(k)]]))
}

pred_mean   <- knn_predict(reg_train["x"], reg_train$y, reg_test["x"], k_reg, mean)
pred_median <- knn_predict(reg_train["x"], reg_train$y, reg_test["x"], k_reg, median)
cat(sprintf("\nRegression (K = %d)\n  MSE: mean %.3f | median %.3f\n  MAE: mean %.3f | median %.3f\n",
            k_reg,
            mean((reg_test$y - pred_mean)^2), mean((reg_test$y - pred_median)^2),
            mean(abs(reg_test$y - pred_mean)), mean(abs(reg_test$y - pred_median))))

xs <- c("x1", "x2")
p_small <- knn_predict(clf_train[xs], clf_train$y, clf_test[xs], k_small)
p_large <- knn_predict(clf_train[xs], clf_train$y, clf_test[xs], k_large)
cat(sprintf("Classification (share y = 1: train %.2f, test %.2f)\n  Accuracy: K = %d %.3f | K = %d %.3f\n  Brier:    K = %d %.3f | K = %d %.3f\n",
            mean(clf_train$y), mean(clf_test$y),
            k_small, mean((p_small > 0.5) == clf_test$y), k_large, mean((p_large > 0.5) == clf_test$y),
            k_small, mean((p_small - clf_test$y)^2),      k_large, mean((p_large - clf_test$y)^2)))
