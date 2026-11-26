# =============================================================================
# Exercise: what you optimise is what you get (MSE vs MAE, accuracy vs Brier)
# Part 2: solution
# -----------------------------------------------------------------------------
# Data (from 01_generate_data.R):
#   data/reg_train.csv, data/reg_test.csv   regression: predictor x, outcome y
#   data/clf_train.csv, data/clf_test.csv   classification: x1, x2, y (0/1),
#                                           balanced classes
# =============================================================================

library(ggplot2)

# ---- Helper: KNN predictions ----------------------------------------------------
# For each case in new_x, find the k nearest cases in train_x (Euclidean
# distance) and summarise their outcomes with `aggregate`.
#   Regression:     aggregate = mean or median
#   Classification: y coded 0/1, aggregate = mean -> share of neighbours with
#                   y = 1, i.e. the predicted probability of y = 1
knn_predict <- function(train_x, train_y, new_x, k, aggregate = mean) {
  train_x <- as.matrix(train_x)
  new_x   <- as.matrix(new_x)
  # squared Euclidean distances: rows = new cases, columns = training cases
  d2 <- outer(rowSums(new_x^2), rowSums(train_x^2), "+") - 2 * new_x %*% t(train_x)
  apply(d2, 1, function(d) aggregate(train_y[order(d)[seq_len(k)]]))
}

# =============================================================================
# Part A: regression, MSE vs MAE
# =============================================================================
reg_train <- read.csv("data/reg_train.csv")
reg_test  <- read.csv("data/reg_test.csv")

# ---- A1: two KNN models with K = 25: neighbour mean vs neighbour median -----------
pred_mean   <- knn_predict(reg_train["x"], reg_train$y, reg_test["x"], k = 25, aggregate = mean)
pred_median <- knn_predict(reg_train["x"], reg_train$y, reg_test["x"], k = 25, aggregate = median)

# ---- A2: evaluate both on the test set with MSE and MAE ----------------------------
mse <- function(obs, pred) mean((obs - pred)^2)
mae <- function(obs, pred) mean(abs(obs - pred))

reg_results <- data.frame(
  model = c("KNN mean", "KNN median"),
  MSE   = c(mse(reg_test$y, pred_mean), mse(reg_test$y, pred_median)),
  MAE   = c(mae(reg_test$y, pred_mean), mae(reg_test$y, pred_median))
)
reg_results
# -> The mean wins on MSE, the median wins on MAE.

# ---- A3: why? Look at the data and both prediction lines ---------------------------
reg_lines <- data.frame(
  x     = rep(reg_test$x, 2),
  pred  = c(pred_mean, pred_median),
  model = rep(c("KNN mean", "KNN median"), each = nrow(reg_test))
)

ggplot(reg_test, aes(x, y)) +
  geom_point(alpha = 0.15, size = 0.8) +
  geom_line(data = reg_lines, aes(x, pred, colour = model), linewidth = 1.2) +
  labs(colour = NULL, title = "Skewed outcome: the mean lies above the median") +
  theme_minimal() +
  theme(legend.position = "bottom")

# The noise is right-skewed: a few cases lie far above the rest.
# - Squared errors punish those large misses heavily, so MSE prefers predictions
#   pulled up towards them: the (conditional) mean.
# - Absolute errors punish all misses in proportion, so MAE prefers the
#   prediction with half of the cases above and half below: the median.
# Which model is "best" depends on which errors matter for the application.

# Residual distributions make the same point
reg_resid <- data.frame(
  residual = c(reg_test$y - pred_mean, reg_test$y - pred_median),
  model    = rep(c("KNN mean", "KNN median"), each = nrow(reg_test))
)
ggplot(reg_resid, aes(residual, fill = model)) +
  geom_histogram(binwidth = 0.25, alpha = 0.6, position = "identity") +
  geom_vline(xintercept = 0) +
  coord_cartesian(xlim = c(-4, 8)) +
  labs(fill = NULL, title = "Residuals of both models",
       subtitle = "Median model: half the residuals above 0. Mean model: residuals average to 0") +
  theme_minimal() +
  theme(legend.position = "bottom")

# =============================================================================
# Part B: classification, accuracy vs Brier score
# =============================================================================
clf_train <- read.csv("data/clf_train.csv")
clf_test  <- read.csv("data/clf_test.csv")
xs <- c("x1", "x2")

# ---- B1: two KNN models: K = 1 vs K = 15 --------------------------------------------
# Predicted probability = share of neighbours with y = 1
prob_k1  <- knn_predict(clf_train[xs], clf_train$y, clf_test[xs], k = 1)
prob_k15 <- knn_predict(clf_train[xs], clf_train$y, clf_test[xs], k = 15)

# ---- B2: evaluate both with accuracy and Brier score ---------------------------------
accuracy <- function(obs, prob) mean((prob > 0.5) == obs)
brier    <- function(obs, prob) mean((prob - obs)^2)   # lower is better

clf_results <- data.frame(
  model    = c("KNN, K = 1", "KNN, K = 15"),
  accuracy = c(accuracy(clf_test$y, prob_k1), accuracy(clf_test$y, prob_k15)),
  brier    = c(brier(clf_test$y, prob_k1),    brier(clf_test$y, prob_k15))
)
clf_results
# -> K = 1 wins on accuracy, K = 15 wins on the Brier score.

# ---- B3: why? Look at the predicted probabilities --------------------------------------
clf_probs <- data.frame(
  prob  = c(prob_k1, prob_k15),
  model = rep(c("KNN, K = 1", "KNN, K = 15"), each = nrow(clf_test))
)
ggplot(clf_probs, aes(prob)) +
  geom_histogram(binwidth = 1 / 15, boundary = 0, fill = "grey40") +
  facet_wrap(~ model) +
  labs(x = "Predicted probability of y = 1", y = "Count",
       title = "K = 1 only predicts 0 or 1: always fully confident") +
  theme_minimal()

# Where in the predictor space does each model win? Map the test cases.
ggplot(clf_test, aes(x1, x2, colour = factor(y))) +
  geom_point(size = 0.6, alpha = 0.6) +
  labs(colour = "y", title = "Test data") +
  coord_equal() +
  theme_minimal()
# -> For x1 < 0 the classes are cleanly separated by a wiggly boundary;
#    for x1 > 0 the classes are completely mixed.

clf_test$region <- ifelse(clf_test$x1 < 0, "x1 < 0 (clear boundary)", "x1 > 0 (no signal)")
by_region <- do.call(rbind, lapply(split(seq_len(nrow(clf_test)), clf_test$region), function(i) {
  data.frame(region = clf_test$region[i[1]],
             model  = c("KNN, K = 1", "KNN, K = 15"),
             accuracy = c(accuracy(clf_test$y[i], prob_k1[i]), accuracy(clf_test$y[i], prob_k15[i])),
             brier    = c(brier(clf_test$y[i], prob_k1[i]),    brier(clf_test$y[i], prob_k15[i])))
}))
rownames(by_region) <- NULL
by_region
# - Clear-boundary region: K = 1 follows the wiggly boundary closely, so it
#   classifies better. K = 15 averages across the boundary and misclassifies
#   more cases near it.
# - No-signal region: both models are at 50% accuracy, since nothing can be
#   predicted. But K = 1 claims 0% or 100% certainty for every case and is
#   wrong half the time, giving the worst possible Brier contribution there
#   (0.5). K = 15 predicts probabilities closer to 0.5, which is honest.
# Accuracy only checks which side of 0.5 a probability falls on; the Brier
# score also checks whether the probability itself is right.

# ---- Optional: same accuracy, worse Brier ----------------------------------------------
# Push K = 15's probabilities to 0 or 1. Every case stays on the same side of
# 0.5, so the predicted classes, and hence the accuracy, are unchanged.
prob_k15_overconfident <- ifelse(prob_k15 > 0.5, 1, 0)

data.frame(
  model    = c("KNN, K = 15", "KNN, K = 15, pushed to 0/1"),
  accuracy = c(accuracy(clf_test$y, prob_k15), accuracy(clf_test$y, prob_k15_overconfident)),
  brier    = c(brier(clf_test$y, prob_k15),    brier(clf_test$y, prob_k15_overconfident))
)
# -> Accuracy cannot tell a model with useful probabilities from one that
#    claims certainty it does not have. The Brier score can: it is a proper
#    scoring rule, rewarding probabilities that match how often things happen.
