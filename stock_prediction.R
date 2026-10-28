
# ============================================================
# PROJECT: Machine Learning-Based Stock Market Direction
# Prediction Using Technical Indicators
# Stock: Reliance Industries (RELIANCE.NS)
# Language: R
# ============================================================

# 1. Load packages
library(dplyr)
library(ggplot2)
library(ranger)
library(xgboost)
library(pROC)

# 2. Create output folders if required
dir.create("outputs", showWarnings = FALSE)
dir.create("report", showWarnings = FALSE)
dir.create("presentation", showWarnings = FALSE)

# 3. Load processed dataset
df <- read.csv("data/model_dataset.csv")
df$Date <- as.Date(df$Date)
df$Direction <- factor(df$Direction, levels = c("Down", "Up"))
df <- df[order(df$Date), ]

features <- c(
  "Return", "Momentum_5", "Momentum_10",
  "MA_Ratio_5", "MA_Ratio_20",
  "Volatility_5", "Volatility_20", "Volume_Change"
)

# Check data
stopifnot(all(features %in% names(df)))
stopifnot(all(c("Date", "Direction") %in% names(df)))

cat("Dataset rows:", nrow(df), "\n")
cat("Date range:", as.character(min(df$Date)), "to",
    as.character(max(df$Date)), "\n")
print(table(df$Direction))

# 4. Chronological 80/20 split
n <- nrow(df)
split_point <- floor(0.80 * n)

train <- df[1:split_point, ]
test <- df[(split_point + 1):n, ]

cat("Training observations:", nrow(train), "\n")
cat("Testing observations:", nrow(test), "\n")
cat("Training period:", as.character(min(train$Date)), "to",
    as.character(max(train$Date)), "\n")
cat("Testing period:", as.character(min(test$Date)), "to",
    as.character(max(test$Date)), "\n")

# 5. Prepare XGBoost matrices
x_train <- as.matrix(train[, features])
x_test <- as.matrix(test[, features])

y_train <- as.integer(train$Direction == "Up")
y_test <- as.integer(test$Direction == "Up")

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

# 6. Logistic Regression
log_model <- glm(
  Direction ~ Return + Momentum_5 + Momentum_10 +
    MA_Ratio_5 + MA_Ratio_20 +
    Volatility_5 + Volatility_20 + Volume_Change,
  data = train,
  family = binomial()
)

log_prob <- predict(log_model, newdata = test, type = "response")
log_pred <- factor(
  ifelse(log_prob >= 0.5, "Up", "Down"),
  levels = c("Down", "Up")
)

# 7. Random Forest
set.seed(42)

rf_model <- ranger(
  Direction ~ Return + Momentum_5 + Momentum_10 +
    MA_Ratio_5 + MA_Ratio_20 +
    Volatility_5 + Volatility_20 + Volume_Change,
  data = train,
  num.trees = 300,
  importance = "permutation",
  probability = TRUE,
  seed = 42
)

rf_prob <- predict(rf_model, data = test)$predictions[, "Up"]
rf_pred <- factor(
  ifelse(rf_prob >= 0.5, "Up", "Down"),
  levels = c("Down", "Up")
)

# 8. XGBoost
set.seed(42)

xgb_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 3,
    eta = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    seed = 42
  ),
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

xgb_prob <- predict(xgb_model, dtest)
xgb_pred <- factor(
  ifelse(xgb_prob >= 0.5, "Up", "Down"),
  levels = c("Down", "Up")
)

# 9. Model evaluation function
evaluate_model <- function(name, actual, predicted, probability) {
  actual <- factor(actual, levels = c("Down", "Up"))
  predicted <- factor(predicted, levels = c("Down", "Up"))
  
  cm <- table(Actual = actual, Predicted = predicted)
  
  tn <- cm["Down", "Down"]
  fp <- cm["Down", "Up"]
  fn <- cm["Up", "Down"]
  tp <- cm["Up", "Up"]
  
  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  
  f1 <- if (is.finite(precision) && is.finite(recall) &&
            (precision + recall) > 0) {
    2 * precision * recall / (precision + recall)
  } else {
    NA_real_
  }
  
  roc_obj <- pROC::roc(
    response = actual,
    predictor = probability,
    levels = c("Down", "Up"),
    direction = "<",
    quiet = TRUE
  )
  
  list(
    metrics = data.frame(
      Model = name,
      Accuracy = mean(actual == predicted),
      Precision_Up = precision,
      Recall_Up = recall,
      F1_Up = f1,
      ROC_AUC = as.numeric(pROC::auc(roc_obj))
    ),
    confusion = cm,
    roc = roc_obj
  )
}

# 10. Evaluate models
results_log <- evaluate_model(
  "Logistic Regression", test$Direction, log_pred, log_prob
)
results_rf <- evaluate_model(
  "Random Forest", test$Direction, rf_pred, rf_prob
)
results_xgb <- evaluate_model(
  "XGBoost", test$Direction, xgb_pred, xgb_prob
)

results <- dplyr::bind_rows(
  results_log$metrics,
  results_rf$metrics,
  results_xgb$metrics
)

# 11. Majority-class baseline
baseline_pred <- factor(
  rep("Up", nrow(test)),
  levels = c("Down", "Up")
)

baseline_accuracy <- mean(baseline_pred == test$Direction)
baseline_cm <- table(
  Actual = test$Direction,
  Predicted = baseline_pred
)

cat("\nBaseline accuracy:", round(baseline_accuracy, 4), "\n")

# 12. Save evaluation results
write.csv(results, "outputs/model_comparison.csv", row.names = FALSE)

write.csv(
  as.data.frame.matrix(results_log$confusion),
  "outputs/logistic_confusion_matrix.csv"
)
write.csv(
  as.data.frame.matrix(results_rf$confusion),
  "outputs/random_forest_confusion_matrix.csv"
)
write.csv(
  as.data.frame.matrix(results_xgb$confusion),
  "outputs/xgboost_confusion_matrix.csv"
)

write.csv(
  as.data.frame.matrix(baseline_cm),
  "outputs/baseline_confusion_matrix.csv"
)

cat("\nModel comparison:\n")
print(transform(
  results,
  Accuracy = round(Accuracy, 4),
  Precision_Up = round(Precision_Up, 4),
  Recall_Up = round(Recall_Up, 4),
  F1_Up = round(F1_Up, 4),
  ROC_AUC = round(ROC_AUC, 4)
))

# 13. Model accuracy chart
accuracy_results <- data.frame(
  Model = c("Baseline", results$Model),
  Accuracy = c(baseline_accuracy, results$Accuracy)
)

accuracy_results$Model <- factor(
  accuracy_results$Model,
  levels = accuracy_results$Model
)

accuracy_plot <- ggplot(
  accuracy_results,
  aes(x = Model, y = Accuracy)
) +
  geom_col(fill = "steelblue") +
  geom_text(
    aes(label = sprintf("%.2f%%", Accuracy * 100)),
    vjust = -0.5
  ) +
  scale_y_continuous(
    limits = c(0, 0.65),
    labels = function(x) paste0(x * 100, "%")
  ) +
  labs(
    title = "Stock Direction Prediction: Model Comparison",
    subtitle = "Chronological test set",
    x = "Model",
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

ggsave(
  "outputs/model_accuracy_comparison.png",
  accuracy_plot,
  width = 9, height = 5, dpi = 300
)

# 14. Random Forest feature importance
importance_values <- ranger::importance(rf_model)

importance_df <- data.frame(
  Feature = names(importance_values),
  Importance = as.numeric(importance_values)
)

write.csv(
  importance_df,
  "outputs/feature_importance.csv",
  row.names = FALSE
)

importance_plot <- ggplot(
  importance_df,
  aes(x = Importance, y = reorder(Feature, Importance))
) +
  geom_col(fill = "darkorange") +
  labs(
    title = "Random Forest: Feature Importance",
    subtitle = "Permutation importance on training data",
    x = "Permutation Importance",
    y = "Technical Indicator"
  ) +
  theme_minimal()

ggsave(
  "outputs/random_forest_feature_importance.png",
  importance_plot,
  width = 9, height = 5, dpi = 300
)

# 15. Historical stock-price chart
if (file.exists("data/stock_prices.csv")) {
  prices <- read.csv("data/stock_prices.csv")
  prices$Date <- as.Date(prices$Date)
  
  price_plot <- ggplot(prices, aes(x = Date, y = Adjusted)) +
    geom_line(linewidth = 0.5, color = "steelblue") +
    labs(
      title = "Historical Adjusted Closing Price of Reliance Industries",
      subtitle = "Daily data obtained from Yahoo Finance",
      x = "Date",
      y = "Adjusted Closing Price (INR)"
    ) +
    theme_minimal()
  
  ggsave(
    "outputs/reliance_stock_price.png",
    price_plot,
    width = 10, height = 5, dpi = 300
  )
}

# 16. Save model summaries
capture.output(
  summary(log_model),
  file = "outputs/logistic_regression_summary.txt"
)

cat("\nProject run completed.\n")
cat("Results and charts saved in the outputs folder.\n")


# Load the processed dataset
df <- read.csv("data/model_dataset.csv")

# Calculate daily return statistics
returns <- df$Return

cat("Number of observations:", length(returns), "\n")
cat("Mean return:", mean(returns), "\n")
cat("Standard deviation:", sd(returns), "\n")
cat("Minimum return:", min(returns), "\n")
cat("25th percentile:", quantile(returns, 0.25), "\n")
cat("Median return:", median(returns), "\n")
cat("75th percentile:", quantile(returns, 0.75), "\n")
cat("Maximum return:", max(returns), "\n")
