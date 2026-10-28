# ============================================================
# DATA PREPARATION
# Project: Stock Direction Prediction Using Machine Learning
# Asset: Reliance Industries (RELIANCE.NS)
#
# Creates:
#   data/stock_prices.csv
#   data/model_dataset.csv
#
# Run this script before stock_prediction.R.
# ============================================================

# 1. Install any missing packages once, if needed:
# install.packages(c("quantmod", "TTR", "dplyr"))

library(quantmod)
library(TTR)
library(dplyr)

dir.create("data", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)

# 2. Download daily stock data from Yahoo Finance.
# The end date in getSymbols is exclusive, so use tomorrow's date
# to include the most recent available trading session.
symbol <- "RELIANCE.NS"
start_date <- as.Date("2021-01-01")
end_date_exclusive <- Sys.Date() + 1

stock_xts <- getSymbols(
  symbol,
  src = "yahoo",
  from = start_date,
  to = end_date_exclusive,
  auto.assign = FALSE
)

if (NROW(stock_xts) == 0) {
  stop("No price data was downloaded. Check the ticker, internet connection, and Yahoo Finance availability.")
}

# 3. Convert OHLCV data to a regular data frame.
# Ad() extracts the adjusted closing price from the Yahoo Finance data.
prices <- data.frame(
  Date = as.Date(index(stock_xts)),
  Open = as.numeric(Op(stock_xts)),
  High = as.numeric(Hi(stock_xts)),
  Low = as.numeric(Lo(stock_xts)),
  Close = as.numeric(Cl(stock_xts)),
  Adjusted = as.numeric(Ad(stock_xts)),
  Volume = as.numeric(Vo(stock_xts))
) %>%
  arrange(Date) %>%
  distinct(Date, .keep_all = TRUE)

write.csv(prices, "data/stock_prices.csv", row.names = FALSE)

# 4. Engineer features and define the next-trading-day direction.
# All predictors for date t use data available through date t.
# Direction for date t is based on whether adjusted close at t+1
# is greater than adjusted close at t.
prices <- prices %>%
  mutate(
    Return = Adjusted / lag(Adjusted) - 1,
    Momentum_5 = Adjusted / lag(Adjusted, 5) - 1,
    Momentum_10 = Adjusted / lag(Adjusted, 10) - 1,
    MA_Ratio_5 = Adjusted / TTR::SMA(Adjusted, n = 5) - 1,
    MA_Ratio_20 = Adjusted / TTR::SMA(Adjusted, n = 20) - 1,
    Volatility_5 = TTR::runSD(Return, n = 5),
    Volatility_20 = TTR::runSD(Return, n = 20),
    Volume_Change = Volume / lag(Volume) - 1,
    Next_Adjusted = lead(Adjusted),
    Direction = case_when(
      is.na(Next_Adjusted) ~ NA_character_,
      Next_Adjusted > Adjusted ~ "Up",
      Next_Adjusted <= Adjusted ~ "Down"
    )
  )

features <- c(
  "Return", "Momentum_5", "Momentum_10",
  "MA_Ratio_5", "MA_Ratio_20",
  "Volatility_5", "Volatility_20", "Volume_Change"
)

# Remove rows where rolling features or next-day target are unavailable.
model_dataset <- prices %>%
  select(Date, all_of(features), Direction) %>%
  filter(if_all(all_of(features), ~ is.finite(.x))) %>%
  filter(!is.na(Direction)) %>%
  arrange(Date)

model_dataset$Direction <- factor(
  model_dataset$Direction,
  levels = c("Down", "Up")
)

# 5. Save the modelling dataset used by stock_prediction.R.
write.csv(model_dataset, "data/model_dataset.csv", row.names = FALSE)

# 6. Print checks for the user.
cat("\nData preparation completed.\n")
cat("Ticker:", symbol, "\n")
cat("Raw price rows:", nrow(prices), "\n")
cat("Model dataset rows:", nrow(model_dataset), "\n")
cat("Model dataset date range:",
    as.character(min(model_dataset$Date)), "to",
    as.character(max(model_dataset$Date)), "\n")
cat("\nDirection counts:\n")
print(table(model_dataset$Direction))
cat("\nSaved files:\n")
cat("- data/stock_prices.csv\n")
cat("- data/model_dataset.csv\n")
cat("\nNext step: run source('stock_prediction.R') from the project root.\n")
