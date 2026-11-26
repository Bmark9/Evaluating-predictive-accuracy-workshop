# =============================================================================
# Exercise: distribution shift and external validation
# Part 2: exercise (worked solution)
# -----------------------------------------------------------------------------
# Data (from 01_generate_data.R), outcome y (continuous):
#   data/internal.csv  development data (n = 1000)
#   data/external.csv  data from a new setting (n = 1000), used once at the end
#
# What we know about the variables:
#   x1, x2, x3  causes of y
#   x4, x5      other measured variables
#   m           a biomarker that rises as a CONSEQUENCE of y
#
# Two candidate models, both random forests:
#   "All predictors": uses everything (x1 ... x5, m)
#   "Causes only":    uses x1, x2, x3
# =============================================================================

library(ranger)
library(ggplot2)

internal <- read.csv("data/internal.csv")
external <- read.csv("data/external.csv")

predictor_sets <- list(
  "All predictors" = c("x1", "x2", "x3", "x4", "x5", "m"),
  "Causes only"    = c("x1", "x2", "x3")
)

# A handful of hyperparameter settings for each model
grid <- expand.grid(mtry = 1:3, min.node.size = c(5, 20, 50))

fit_rf <- function(data, vars, mtry, min.node.size) {
  ranger(reformulate(vars, response = "y"), data = data,
         mtry = mtry, min.node.size = min.node.size)
}
mse <- function(fit, data) mean((data$y - predict(fit, data)$predictions)^2)

# ---- Step 1: split the internal data: train 60% / validation 20% / test 20% ------------
set.seed(2026)
set_label <- sample(rep(c("train", "validation", "test"), times = c(600, 200, 200)))
train      <- internal[set_label == "train", ]
validation <- internal[set_label == "validation", ]
test       <- internal[set_label == "test", ]

# ---- Step 2: tune each model on the validation set ------------------------------------------
tuning <- do.call(rbind, lapply(names(predictor_sets), function(model) {
  vars <- predictor_sets[[model]]
  val_mse <- mapply(function(mtry, nodesize) {
    mse(fit_rf(train, vars, mtry, nodesize), validation)
  }, grid$mtry, grid$min.node.size)
  data.frame(model = model, grid, validation_mse = val_mse)
}))
tuning

best <- do.call(rbind, lapply(split(tuning, tuning$model), function(t) t[which.min(t$validation_mse), ]))
rownames(best) <- NULL
best

# ---- Step 3: internal validation on the (untouched) internal test set ------------------------
best$internal_test_mse <- mapply(function(model, mtry, nodesize) {
  mse(fit_rf(train, predictor_sets[[model]], mtry, nodesize), test)
}, best$model, best$mtry, best$min.node.size)

best[, c("model", "mtry", "min.node.size", "internal_test_mse")]
# -> Internally, the all-predictors model is far better: the biomarker m
#    tracks y closely. Based on internal validation, it is the clear choice.

# ---- Step 4: refit on ALL internal data, then validate externally -----------------------------
final_models <- lapply(setNames(nm = best$model), function(model) {
  b <- best[best$model == model, ]
  fit_rf(internal, predictor_sets[[model]], b$mtry, b$min.node.size)
})

best$external_mse <- sapply(best$model, function(model) mse(final_models[[model]], external))
best[, c("model", "internal_test_mse", "external_mse")]
# -> Externally the ranking flips: the all-predictors model gets much worse,
#    the causes-only model performs as well as internally.

results_long <- rbind(
  data.frame(model = best$model, validation = "Internal (test set)", mse = best$internal_test_mse),
  data.frame(model = best$model, validation = "External",            mse = best$external_mse)
)
results_long$validation <- factor(results_long$validation,
                                  levels = c("Internal (test set)", "External"))

ggplot(results_long, aes(validation, mse, colour = model, group = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  labs(x = NULL, y = "MSE", colour = NULL,
       title = "Internal winner, external loser") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")

# ---- Step 5: why? -----------------------------------------------------------------------------
# What does the all-predictors model rely on?
imp_fit <- ranger(y ~ x1 + x2 + x3 + x4 + x5 + m, data = internal,
                  mtry = best$mtry[best$model == "All predictors"],
                  min.node.size = best$min.node.size[best$model == "All predictors"],
                  importance = "permutation")
sort(imp_fit$variable.importance, decreasing = TRUE)
# -> Mostly on m.

# Has the relationship between m and y changed?
marker <- rbind(data.frame(setting = "Internal", internal[c("m", "y")]),
                data.frame(setting = "External", external[c("m", "y")]))
ggplot(marker, aes(y, m, colour = setting)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1.2) +
  labs(colour = NULL, title = "The biomarker behaves differently in the new setting") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom")

# And the causes? Same distribution and same relationship with y in both settings.
summary(internal[c("x1", "x2", "x3")])
summary(external[c("x1", "x2", "x3")])

# Take-away:
# - Internal validation was done properly (tuning on validation, honest test
#   set), and it correctly said the all-predictors model is better HERE.
# - It cannot tell you whether that holds ELSEWHERE. The all-predictors model
#   exploits a relationship (m with y) that is specific to the development
#   setting. The causal relationships carried over; the non-causal one did not.
# - External validation reveals this. Causal knowledge helps anticipate it.
