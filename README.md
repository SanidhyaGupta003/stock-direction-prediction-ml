# Stock Direction Prediction Using Machine Learning

## Overview

This project investigates whether technical indicators derived from historical stock market data can help predict the next trading day's price direction for Reliance Industries Limited (`RELIANCE.NS`).

Three classification models are implemented and compared using R:

- Logistic Regression
- Random Forest
- XGBoost

The models are evaluated on a chronological train-test split using classification metrics and an always-Up baseline.

## Objectives

- Develop a machine learning pipeline for next-day stock direction classification.
- Engineer technical indicators from historical market data.
- Compare linear and tree-based classification models.
- Evaluate performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- Examine the limitations of short-horizon stock direction prediction.

## Dataset

- **Asset:** Reliance Industries Limited (`RELIANCE.NS`)
- **Data source:** Yahoo Finance
- **Period:** January 2021 onward, subject to available observations
- **Target:** Next trading day's direction (`Up` or `Down`)

The modelling dataset contains eight engineered features:

| Feature | Description |
|---|---|
| `Return` | Daily adjusted-price return |
| `Momentum_5` | Five-period momentum |
| `Momentum_10` | Ten-period momentum |
| `MA_Ratio_5` | Price deviation from the five-period moving average |
| `MA_Ratio_20` | Price deviation from the twenty-period moving average |
| `Volatility_5` | Five-period rolling return volatility |
| `Volatility_20` | Twenty-period rolling return volatility |
| `Volume_Change` | Relative change in trading volume |

## Methodology

1. Obtain historical stock data from Yahoo Finance.
2. Prepare the modelling dataset and calculate technical features.
3. Sort observations chronologically and use an 80:20 train-test split.
4. Train Logistic Regression, Random Forest, and XGBoost.
5. Evaluate predictions on the held-out test set.
6. Compare the models with an always-Up baseline.

The reported experiment used 1,130 training observations and 283 test observations.

## Results

The following metrics are from the reported test-set evaluation.

| Model | Accuracy | Precision (Up) | Recall (Up) | F1-score (Up) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 51.59% | 50.26% | 68.84% | 58.10% | 0.5384 |
| Random Forest | **52.65%** | 51.18% | 63.04% | 56.49% | 0.5014 |
| XGBoost | 47.70% | 46.79% | 52.90% | 49.66% | 0.4846 |
| Always-Up baseline | 48.76% | — | — | — | — |

Random Forest achieved the highest accuracy among the three trained models. Its accuracy was approximately 3.89 percentage points above the always-Up baseline. However, its ROC-AUC was close to 0.5, indicating limited discrimination between upward and downward movements.

These results do not establish that the model can generate profitable trades.

## Technologies

- R
- `quantmod`
- `TTR`
- `dplyr`
- `ggplot2`
- `ranger`
- `xgboost`
- `pROC`

## Project Structure

```text
stock-direction-prediction-ml/
├── stock_prediction.R
├── README.md
├── data/
│   └── model_dataset.csv
└── outputs/
    ├── model_comparison.csv
    ├── model_accuracy_comparison.png
    ├── feature_importance.csv
    ├── logistic_confusion_matrix.csv
    ├── random_forest_confusion_matrix.csv
    └── xgboost_confusion_matrix.csv
```

The structure above describes the intended project organization. Include only files that are actually present in the repository.

## How to Run

1. Install R and RStudio.
2. Install the required packages:

```r
install.packages(c(
  "quantmod", "TTR", "dplyr", "ggplot2",
  "ranger", "xgboost", "pROC", "tidyr"
))
```

3. Ensure the required processed dataset is available at `data/model_dataset.csv`.
4. Set the working directory to the project root.
5. Run the modelling script:

```r
source("stock_prediction.R")
```

The script expects the processed modelling dataset to exist before execution. Ensure the required input file and folder structure are available before running it.

## Limitations

- The experiment focuses on one stock.
- The test results show limited predictive performance.
- Technical indicators alone may not capture all factors affecting stock prices.
- Classification accuracy does not measure transaction costs, portfolio returns, or trading profitability.
- Further evaluation across assets, time periods, and market conditions is needed.

## Future Work

- Extend the analysis to multiple assets.
- Evaluate walk-forward validation and alternative decision thresholds.
- Investigate additional market features and feature-selection methods.
- Add transaction-cost-aware backtesting and portfolio-level risk metrics.
- Compare model performance across different market regimes.

## Disclaimer

This project is for educational and research purposes only. It is not investment advice and does not guarantee future market performance or trading profits.
