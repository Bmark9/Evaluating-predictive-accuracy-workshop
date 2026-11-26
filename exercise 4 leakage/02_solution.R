# =============================================================================
# Exercise: tuning on the validation set -> over-optimistic accuracy
# Part 2: solution
# -----------------------------------------------------------------------------
# Data (from 01_generate_data.R), outcome y is binary with balanced classes:
#   data/knn_train.csv       fit the KNN model
#   data/knn_validation.csv  tune K
#   data/knn_test.csv        estimate accuracy on new data
# =============================================================================

library(class)    # knn()
library(ggplot2)

train      <- read.csv("data/knn_train.csv")
validation <- read.csv("data/knn_validation.csv")
test       <- read.csv("data/knn_test.csv")

predictors <- setdiff(names(train), "y")
ks <- seq(1, 30, by = 1)   # odd K: no tied votes with two classes

# Accuracy of a KNN model fitted on `fit_data`, evaluated on `new_data`
knn_accuracy <- function(k, fit_data, new_data) {
  pred <- knn(train = fit_data[, predictors], test = new_data[, predictors],
              cl = factor(fit_data$y), k = k)
  mean(pred == new_data$y)
}

# ---- Step 1: validation accuracy for each K -------------------------------------
val_acc <- sapply(ks, knn_accuracy, fit_data = train, new_data = validation)

results <- data.frame(K = ks, validation = val_acc)

ggplot(results, aes(K, validation)) +
  geom_line() + geom_point() +
  labs(y = "Validation accuracy", title = "Validation accuracy across K") +
  theme_minimal()

# ---- Step 2: test accuracy for each K ---------------------------------------------
results$test <- sapply(ks, knn_accuracy, fit_data = train, new_data = test)

results_long <- rbind(
  data.frame(K = ks, set = "Validation", accuracy = results$validation),
  data.frame(K = ks, set = "Test",       accuracy = results$test)
)

ggplot(results_long, aes(K, accuracy, colour = set)) +
  geom_line() + geom_point() +
  labs(y = "Accuracy", colour = NULL,
       title = "Validation vs test accuracy across K") +
  theme_minimal()

# ---- Step 3: differences between validation and test accuracy ----------------------
results$difference <- results$validation - results$test

mean(results$difference)   # close to 0: on average, validation accuracy is fair
sd(results$difference)     # but each single K has a noisy estimate

ggplot(results, aes(K, difference)) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_hline(yintercept = mean(results$difference), linetype = "dashed") +
  geom_col(fill = "grey60") +
  labs(y = "Validation - test accuracy",
       title = "For any single K, the difference is noise around zero",
       subtitle = "Dashed line: mean difference across all K") +
  theme_minimal()

# ---- Step 4: pick K with the highest validation accuracy ---------------------------
best <- which.max(results$validation)   # first (smallest) K if tied
results[best, ]

results[best, "difference"] / sd(results$difference)

# The difference for the selected K is far larger than the typical
# difference across K: picking the maximum also picks up the luckiest
# validation estimate. The validation accuracy of the selected K is no
# longer an honest estimate of how well the model does on new data.

# How much of this is a bad choice of K, and how much a bad estimate?
max(results$test)                        # best K we could have picked

results$test[best]                       # test accuracy of the K we picked
# -> The chosen K is nearly as good as the best K; the problem is mostly
#    the reported accuracy, not the choice.

