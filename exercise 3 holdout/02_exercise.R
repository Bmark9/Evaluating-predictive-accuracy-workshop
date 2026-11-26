# =============================================================================
# Exercise: how big should the hold-out set be? (random forest vs Lasso)
# Part 2: exercise (worked solution)
# -----------------------------------------------------------------------------
# Data (from 01_generate_data.R):
#   data/holdout_data.csv  n = 500: your data. Outcome y, predictors x1 ... x20
#   data/external.csv      n = 10000: "the future". Normally you would not have
#                          this; here it shows the TRUE performance of a model.
#
# Task: for 50 hold-out sizes (5% to 60% of the data), make ONE random split,
# fit a random forest and a Lasso regression on the training part, and record
#   - the hold-out MSE (what you would report)
#   - the external MSE (what you would really get)
# Then compare: how stable are the hold-out estimates, how good are the models,
# and which model would you have picked?
# =============================================================================

library(ranger)    # random forest
library(glmnet)    # Lasso
library(ggplot2)

dat      <- read.csv("data/holdout_data.csv")
external <- read.csv("data/external.csv")
predictors <- setdiff(names(dat), "y")

# ---- Helpers: fit both models, compute MSE of both --------------------------------
# Random forest: default settings (500 trees).
# Lasso: penalty chosen by 10-fold cross-validation within the training data.
fit_models <- function(train) {
  list(
    rf    = ranger(y ~ ., data = train),
    lasso = cv.glmnet(as.matrix(train[predictors]), train$y)
  )
}

mse_models <- function(models, newdata) {
  pred_rf    <- predict(models$rf, newdata)$predictions
  pred_lasso <- predict(models$lasso, as.matrix(newdata[predictors]), s = "lambda.1se")
  c(rf    = mean((newdata$y - pred_rf)^2),
    lasso = mean((newdata$y - pred_lasso)^2))
}

# ---- Step 1: one split per hold-out size ---------------------------------------------
set.seed(2026)
holdout_shares <- seq(0.05, 0.60, length.out = 50)

results <- do.call(rbind, lapply(holdout_shares, function(share) {
  n_holdout <- round(share * nrow(dat))
  idx       <- sample(nrow(dat), n_holdout)
  train     <- dat[-idx, ]
  holdout   <- dat[idx, ]

  models <- fit_models(train)
  mse_holdout  <- mse_models(models, holdout)
  mse_external <- mse_models(models, external)

  data.frame(share   = share,
             model   = c("Random forest", "Lasso"),
             holdout = unname(mse_holdout),
             external = unname(mse_external))
}))   # takes ~30 seconds

head(results)

# ---- Step 2: the hold-out estimates ------------------------------------------------------
ggplot(results, aes(share, holdout, colour = model)) +
  geom_line() + geom_point(size = 1) +
  labs(x = "Share of data held out", y = "Hold-out MSE", colour = NULL,
       title = "Hold-out estimates: one random split per size") +
  theme_minimal() +
  theme(legend.position = "bottom")
# Q: Where are the estimates most erratic?
# -> With small hold-out sets: an MSE computed on 25-50 cases depends heavily
#    on which cases happen to be held out.

# ---- Step 3: the true performance of the same models ---------------------------------------
ggplot(results, aes(share, external, colour = model)) +
  geom_line() + geom_point(size = 1) +
  labs(x = "Share of data held out", y = "External MSE (true performance)", colour = NULL,
       title = "True performance: the more you hold out, the worse the model") +
  theme_minimal() +
  theme(legend.position = "bottom")
# Q: How does true performance change with the hold-out size, for each model?
# -> Holding out more leaves less data for training, so both models get worse.
#    The random forest suffers much more: as a flexible model it needs more
#    data. Lasso hardly changes. With a large hold-out set the lines cross.

# ---- Step 4: which model would you pick? ------------------------------------------------------
# The model you finally use is refitted on ALL data. Which one is really better?
final_models <- fit_models(dat)
mse_models(final_models, external)
# -> Trained on all 500 cases, the random forest is better.

picks <- do.call(rbind, lapply(split(results, results$share), function(r) {
  data.frame(share = r$share[1],
             pick_holdout  = r$model[which.min(r$holdout)],
             pick_external = r$model[which.min(r$external)])
}))

ggplot(picks, aes(share, pick_holdout)) +
  geom_point(size = 2.5) +
  labs(x = "Share of data held out", y = NULL,
       title = "Which model wins on the hold-out set?") +
  theme_minimal()
# Q: Would you have picked the right model? Does it depend on the hold-out size?
# -> With a small hold-out set the pick is a lottery: the estimates are noisy.
#    With a large hold-out set the pick is systematically wrong: the random
#    forest is evaluated after being trained on too little data, so Lasso
#    looks better even though the random forest is better when trained on
#    all data.

# Summary: a hold-out set trades the precision of the estimate (bigger is
# better) against the quality of the model being evaluated (smaller is
# better), and flexible models pay more for lost training data.
# Cross-validation softens this trade-off: every case is used for both.
