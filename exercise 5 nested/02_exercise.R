# =============================================================================
# Exercise: an honest test set after tuning on the validation set
# Part 2: exercise (worked solution)
# -----------------------------------------------------------------------------
# Data (from 01_generate_data.R), outcome y is binary with balanced classes:
#   data/knn_train.csv         fit the KNN model
#   data/knn_validation.csv    tune K
#   data/knn_holdout_test.csv  evaluate the chosen model, once (60 cases)
#   data/knn_population.csv    "the truth": normally you would not have this;
#                              here it reveals the model's true accuracy
# =============================================================================

library(class)    # knn()
library(ggplot2)

train        <- read.csv("data/knn_train.csv")
validation   <- read.csv("data/knn_validation.csv")
holdout_test <- read.csv("data/knn_holdout_test.csv")
population   <- read.csv("data/knn_population.csv")

predictors <- setdiff(names(train), "y")
ks <- 1:30

# Accuracy of a KNN model fitted on `fit_data`, evaluated on `new_data`
knn_accuracy <- function(k, fit_data, new_data) {
  pred <- knn(train = fit_data[, predictors], test = new_data[, predictors],
              cl = factor(fit_data$y), k = k)
  mean(pred == new_data$y)
}

# 95% confidence interval for an accuracy estimated on n cases
# (normal approximation: estimate +/- 1.96 standard errors)
accuracy_ci <- function(acc, n) {
  se <- sqrt(acc * (1 - acc) / n)
  c(lower = acc - 1.96 * se, upper = acc + 1.96 * se)
}

set.seed(1)   # knn() breaks tied votes (even K) at random

# ---- Step 1: tune K on the validation set (as in the previous exercise) -------------
val_acc <- sapply(ks, knn_accuracy, fit_data = train, new_data = validation)

best_k   <- ks[which.max(val_acc)]   # first (smallest) K if tied
best_val <- max(val_acc)
best_k
best_val

# ---- Step 2: evaluate the chosen model on the hold-out test set ----------------------
# The test set is used once, for the chosen K only.
# (Common practice: first refit the chosen model on train + validation. We keep
#  the model fitted on train only, so all estimates refer to the same model.)
test_acc <- knn_accuracy(best_k, fit_data = train, new_data = holdout_test)
test_acc
accuracy_ci(test_acc, n = nrow(holdout_test))

# ---- Step 3: compare with the true accuracy --------------------------------------------
true_acc <- knn_accuracy(best_k, fit_data = train, new_data = population)
true_acc

estimates <- data.frame(
  source   = c("Validation set\n(used to choose K)", "Hold-out test set\n(untouched)"),
  accuracy = c(best_val, test_acc),
  rbind(accuracy_ci(best_val, nrow(validation)),
        accuracy_ci(test_acc, nrow(holdout_test)))
)
estimates$source <- factor(estimates$source, levels = estimates$source)
estimates

ggplot(estimates, aes(source, accuracy)) +
  geom_hline(yintercept = true_acc, linetype = "dashed") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.12) +
  geom_point(size = 3) +
  annotate("text", x = 0.5, y = true_acc, label = "True accuracy",
           hjust = 0, vjust = -0.6) +
  labs(x = NULL, y = "Accuracy (with 95% CI)",
       title = sprintf("Accuracy of the selected model (K = %d)", best_k)) +
  theme_minimal(base_size = 14)

# Q: Which estimate would you report?
# -> The test estimate. Both sets have 60 cases, so both estimates are
#    equally precise (similar CI widths). But the validation estimate is
#    optimistic: K was chosen because it scored best there, partly by luck.
#    Its CI even misses the true accuracy. The test set played no part in
#    the choice, so its estimate is honest: close to the truth here, and
#    right on average (see optional part).
# Note: the test set is only honest the FIRST time. If you go back and
# change the model after seeing the test result, it becomes another
# validation set.

# ---- Optional: is the test estimate right on average? ------------------------------------
# Imagine drawing many different test sets of 60 cases. Each gives a different
# estimate; are they centred on the truth, and do the CIs cover it ~95% of the time?
set.seed(2)
pred_pop <- knn(train[, predictors], population[, predictors],
                cl = factor(train$y), k = best_k)
correct  <- pred_pop == population$y

sim_test <- replicate(2000, {
  i   <- sample(nrow(population), nrow(holdout_test))
  acc <- mean(correct[i])
  ci  <- accuracy_ci(acc, length(i))
  c(acc = acc, covers = ci[["lower"]] <= true_acc & true_acc <= ci[["upper"]])
})

mean(sim_test["acc", ])      # average test estimate: close to the true accuracy
mean(sim_test["covers", ])   # share of CIs covering the truth: close to 95%

ggplot(data.frame(acc = sim_test["acc", ]), aes(acc)) +
  geom_histogram(binwidth = 1 / nrow(holdout_test), fill = "grey60", colour = "white") +
  geom_vline(xintercept = true_acc, linetype = "dashed") +
  geom_vline(xintercept = best_val, colour = "firebrick", linewidth = 1) +
  annotate("text", x = best_val, y = Inf, label = "Validation\nestimate",
           colour = "firebrick", hjust = -0.1, vjust = 1.2) +
  labs(x = "Test accuracy (60 cases)", y = "Number of test sets",
       title = "Test estimates from 2000 possible test sets",
       subtitle = "Dashed: true accuracy") +
  theme_minimal(base_size = 14)
mean(sim_test["acc", ] >= best_val)   # share of test sets scoring as high as the validation estimate
# -> Test estimates scatter around the truth (unbiased, but noisy with 60 cases);
#    the validation estimate lies in the upper tail: only ~4% of honest test
#    sets would score that high.
