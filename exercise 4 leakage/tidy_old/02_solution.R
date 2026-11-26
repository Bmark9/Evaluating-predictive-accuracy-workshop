# =============================================================================
# Exercise 2
# -----------------------------------------------------------------------------
#   Outcome y is binary with balanced classes:
#   data/knn_train.csv       fit the KNN model
#   data/knn_validation.csv  tune K
#   data/knn_test.csv        estimate accuracy on new data
# =============================================================================

library(tidymodels)   
require("kknn") # KNN engine needs installing

# load datasets
read_data <- function(file) read.csv(file) |> mutate(y = factor(y))
train      <- read_data("data/knn_train.csv")
validation <- read_data("data/knn_validation.csv")
test       <- read_data("data/knn_test.csv")

# ---- Model: KNN with K to be tuned ---------------------------------------------
knn_spec <- nearest_neighbor(neighbors = tune(), weight_func = "rectangular") |>
  set_engine("kknn") |>
  set_mode("classification")

knn_wf <- workflow() |> # a tidymodels workflow
  add_formula(y ~ .) |> # formula: y is outcome; all other variables are predictors
  add_model(knn_spec) # # specifies a model

k_grid <- tibble(neighbors = seq(1, 25, by = 1))   # grid of odd K values for the model

# ---- Step 1: validation accuracy for each K --------------------------------------
# A single split: fit on train, assess on validation
val_split <- manual_rset(list(make_splits(train, assessment = validation)),
                         ids = "validation") # specify the train & validation datasets

val_res <- tune_grid(knn_wf, resamples = val_split, grid = k_grid,
                     metrics = metric_set(accuracy)) # fit & evaluate across k_grid

results <- collect_metrics(val_res) |>
  select(K = neighbors, validation = mean) # create a dataset of K & validation accuracy

# make a plot of validation accuracy against K
ggplot(results, aes(K, validation)) +
  geom_line() + geom_point() +
  labs(y = "Validation accuracy", title = "Validation accuracy across K") +
  theme_minimal()

# ---- Step 2: test accuracy for each K ---------------------------------------------
# In practice you would NOT do this: the test set is for the final model only.
# Here we do it to see what the validation set cannot tell us.
test_split <- manual_rset(list(make_splits(train, assessment = test)),
                          ids = "test") # specify the train & test datasets

test_res <- tune_grid(knn_wf, resamples = test_split, grid = k_grid,
                      metrics = metric_set(accuracy)) # fit & evaluate across k_grid (takes a moment)

results <- results |>
  left_join(collect_metrics(test_res) |> select(K = neighbors, test = mean),
            by = "K") # add test performance to results table

results_long <- results |>
  pivot_longer(c(validation, test), names_to = "set", values_to = "accuracy")

ggplot(results_long, aes(K, accuracy, colour = set)) +
  geom_line() + geom_point() +
  labs(y = "Accuracy", colour = NULL,
       title = "Validation vs test accuracy across K") +
  theme_minimal()

# ---- Step 3: differences between validation and test accuracy ----------------------
results <- results |> mutate(difference = validation - test)

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
best_k <- select_best(val_res, metric = "accuracy")
best_k

# Accuracy we would report based on tuning
show_best(val_res, metric = "accuracy", n = 1)

# Final model: fit on train with the selected K, evaluate once on test
final_fit <- knn_wf |>
  finalize_workflow(best_k) |>
  last_fit(split = make_splits(train, assessment = test),
           metrics = metric_set(accuracy))
collect_metrics(final_fit)

results |> filter(K == best_k$neighbors)

# The difference for the selected K is far larger than the typical
# difference across K: picking the maximum also picks up the luckiest
# validation estimate. The validation accuracy of the selected K is no
# longer an honest estimate of how well the model does on new data.

# How much of this is a bad choice of K, and how much a bad estimate?
results |> slice_max(test, n = 1)   # best K we could have picked
# -> The chosen K is nearly as good as the best K; the problem is mostly
#    the reported accuracy, not the choice.

sel <- results |> filter(K == best_k$neighbors)
ggplot(results_long, aes(K, accuracy, colour = set)) +
  geom_line() + geom_point() +
  geom_vline(xintercept = sel$K, linetype = "dashed") +
  annotate("segment", x = sel$K, xend = sel$K, y = sel$test, yend = sel$validation,
           colour = "black", linewidth = 1.2,
           arrow = arrow(ends = "both", length = unit(0.08, "inches"))) +
  labs(y = "Accuracy", colour = NULL,
       title = sprintf("Selected K = %d: validation %.3f vs test %.3f",
                       sel$K, sel$validation, sel$test)) +
  theme_minimal()
