# =============================================================================
# Exercise: how big should the hold-out set be? (random forest vs Lasso)
# Part 3 (instructor): full curves from many splits per hold-out size
# -----------------------------------------------------------------------------
# Same data and models as 02_exercise.R, but with many random splits per
# hold-out size instead of one, so the curves show the systematic pattern:
#   1. Hold-out estimates: spread (noise) shrinks, mean (pessimism) rises
#   2. True performance of the trained models: learning curves cross
#   3. Error of the hold-out estimate vs the model you deploy: U-shaped
#   4. How often the hold-out set picks the right model
#
# Runtime: 12 sizes x 100 splits x 2 models; ~12 minutes on a 2-core machine,
# faster with more cores (ranger uses all). Lower n_reps for a quicker version.
# =============================================================================

library(ranger)
library(glmnet)
library(ggplot2)

dat      <- read.csv("data/holdout_data.csv")
external <- read.csv("data/external.csv")
predictors <- setdiff(names(dat), "y")

holdout_shares <- seq(0.05, 0.60, by = 0.05)
n_reps         <- 100     # random splits per hold-out size

# ---- Helpers (identical to 02_exercise.R) -------------------------------------------
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

model_labels <- c(rf = "Random forest", lasso = "Lasso")

# ---- Reference: the models you would deploy (trained on all data) --------------------
set.seed(1)
deployed <- mse_models(fit_models(dat), external)
deployed
best_model <- names(which.min(deployed))   # the right pick
deployed_df <- data.frame(model = model_labels[names(deployed)], mse = unname(deployed))

# ---- Many splits per hold-out size --------------------------------------------------------
set.seed(2026)
sims <- do.call(rbind, lapply(holdout_shares, function(share) {
  message("Hold-out share ", share)
  do.call(rbind, lapply(seq_len(n_reps), function(rep) {
    idx    <- sample(nrow(dat), round(share * nrow(dat)))
    models <- fit_models(dat[-idx, ])
    ho     <- mse_models(models, dat[idx, ])
    ex     <- mse_models(models, external)
    data.frame(share = share, rep = rep,
               model = model_labels[names(ho)],
               holdout = unname(ho), external = unname(ex),
               correct_pick = names(which.min(ho)) == best_model)
  }))
}))

summ <- do.call(rbind, lapply(split(sims, list(sims$share, sims$model)), function(s) {
  target <- deployed[names(model_labels)[model_labels == s$model[1]]]
  data.frame(share = s$share[1], model = s$model[1],
             ho_mean  = mean(s$holdout),
             ho_lo    = quantile(s$holdout, 0.10),
             ho_hi    = quantile(s$holdout, 0.90),
             ex_mean  = mean(s$external),
             # error of the hold-out estimate as an estimate of the deployed
             # model's true MSE: combines noise (spread) and pessimism (bias)
             rmse_estimate = sqrt(mean((s$holdout - target)^2)))
}))
rownames(summ) <- NULL

picks <- aggregate(correct_pick ~ share, data = sims[!duplicated(sims[c("share", "rep")]), ],
                   FUN = mean)

theme_set(theme_minimal(base_size = 14) + theme(legend.position = "bottom"))

# ---- 1. Hold-out estimates: noise and pessimism ---------------------------------------------
ggplot(summ, aes(share, ho_mean, colour = model, fill = model)) +
  geom_ribbon(aes(ymin = ho_lo, ymax = ho_hi), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1) +
  geom_hline(data = deployed_df, aes(yintercept = mse, colour = model),
             linetype = "dashed") +
  labs(x = "Share of data held out", y = "Hold-out MSE", colour = NULL, fill = NULL,
       title = "Hold-out estimates: mean and 80% range across splits",
       subtitle = "Dashed: true MSE of the model trained on all data (what you deploy)")

# ---- 2. True performance of the trained models ----------------------------------------------
ggplot(summ, aes(share, ex_mean, colour = model)) +
  geom_line(linewidth = 1) + geom_point() +
  labs(x = "Share of data held out", y = "External MSE (true performance)", colour = NULL,
       title = "The random forest pays more for lost training data")

# ---- 3. How far off is the hold-out estimate? ------------------------------------------------
best_share <- do.call(rbind, lapply(split(summ, summ$model), function(s) s[which.min(s$rmse_estimate), ]))
ggplot(summ, aes(share, rmse_estimate, colour = model)) +
  geom_line(linewidth = 1) + geom_point() +
  geom_point(data = best_share, size = 5, shape = 21, stroke = 1.5, fill = NA) +
  labs(x = "Share of data held out",
       y = "Typical error of the estimate (RMSE)", colour = NULL,
       title = "Noisy when small, pessimistic when large",
       subtitle = "Error of hold-out MSE vs true MSE of the deployed model (circle = minimum)")

# ---- 4. Would you pick the right model? -------------------------------------------------------
ggplot(picks, aes(share, correct_pick)) +
  geom_hline(yintercept = 0.5, colour = "grey60", linetype = "dotted") +
  geom_line(linewidth = 1) + geom_point() +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(x = "Share of data held out", y = "Splits picking the right model",
       title = sprintf("How often the hold-out set picks the %s",
                       tolower(model_labels[best_model])),
       subtitle = "Right model = best when trained on all data")
