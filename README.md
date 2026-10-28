# Stock Direction Prediction Using Machine Learning

## Overview

This project investigates whether technical indicators derived from historical stock market data can help predict the next trading day's price direction for Reliance Industries Limited (`RELIANCE.NS`).

Three machine learning classification models are implemented and compared using R:

- Logistic Regression
- Random Forest
- XGBoost

The models are evaluated using a chronological train-test split and compared against an always-Up baseline.

## Objectives

- Develop a machine learning pipeline for next-day stock direction classification.
- Obtain historical stock market data and engineer technical indicators.
- Compare linear and tree-based classification models.
- Evaluate model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- Examine the limitations of short-horizon stock direction prediction.

## Dataset

- **Asset:** Reliance Industries Limited (`RELIANCE.NS`)
- **Data source:** Yahoo Finance
- **Start date:** January 2021
- **End date:** Determined by the latest available trading data when the preparation script runs
- **Target variable:** Next trading day's direction (`Up` or `Down`)

The data-preparation script downloads daily price data and creates eight predictor features.

| Feature | Description |
|---|---|
| `Return` | Daily adjusted-price return |
| `Momentum_5` | Five-trading-period momentum |
| `Momentum_10` | Ten-trading-period momentum |
| `MA_Ratio_5` | Adjusted price relative to the five-period moving average |
| `MA_Ratio_20` | Adjusted price relative to the twenty-period moving average |
| `Volatility_5` | Five-period rolling standard deviation of returns |
| `Volatility_20` | Twenty-period rolling standard deviation of returns |
| `Volume_Change` | Relative change in trading volume |

The target is created by comparing the adjusted closing price on a trading day with the adjusted closing price on the following trading day.

## Methodology

The project follows these steps:

1. Download historical daily stock data from Yahoo Finance.
2. Calculate returns, momentum, moving-average ratios, volatility, and volume changes.
3. Create the next-trading-day direction target.
4. Remove observations with unavailable feature values or target labels.
5. Sort observations chronologically.
6. Split the dataset into training and testing sets using an 80:20 chronological split.
7. Train Logistic Regression, Random Forest, and XGBoost models.
8. Evaluate the models on the held-out test set.
9. Compare model performance against an always-Up baseline.

### Model Configuration

- **Logistic Regression:** Binomial generalized linear model.
- **Random Forest:** 300 trees with permutation feature importance and probability predictions.
- **XGBoost:** Binary logistic objective, maximum tree depth of 3, learning rate of 0.05, and 100 boosting rounds.

The classification threshold for the predicted Up probability is 0.5 for all three models.

## Results

The following results are from the latest reported experiment using a chronological 80:20 train-test split.

- **Total modelling observations:** 1,416
- **Training observations:** 1,132
- **Testing observations:** 284
- **Training period:** February 1, 2021, to September 1, 2025
- **Testing period:** September 2, 2025, to October 27, 2026

### Model Comparison

| Model | Accuracy | Precision (Up) | Recall (Up) | F1-score (Up) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **52.46%** | 50.79% | 69.57% | 58.72% | 0.5363 |
| Random Forest | 52.11% | 50.57% | 63.77% | 56.41% | 0.4967 |
| XGBoost | 49.30% | 48.12% | 55.80% | 51.68% | 0.4599 |
| Always-Up baseline | 48.59% | — | — | — | — |

Logistic Regression achieved the highest test accuracy and ROC-AUC among the three trained models in this experiment. Its accuracy was approximately 3.87 percentage points above the always-Up baseline.

However, all three models exhibited limited predictive performance. The Random Forest and XGBoost ROC-AUC values were close to or below 0.5, while Logistic Regression achieved a ROC-AUC of 0.5363. These results suggest that the tested technical indicators provide limited predictive information for next-day stock direction in this experiment.

Classification accuracy alone does not establish whether a model can generate profitable trades.

## Technologies Used

- **Language:** R
- **Market data:** Yahoo Finance
- **Data retrieval:** `quantmod`
- **Technical indicators:** `TTR`
- **Data manipulation:** `dplyr`
- **Visualization:** `ggplot2`
- **Random Forest:** `ranger`
- **Gradient boosting:** `xgboost`
- **Classification evaluation:** `pROC`

## Repository Structure

```text
stock-direction-prediction-ml/
├── README.md
├── data_preparation.R
├── stock_prediction.R
├── data/
│   ├── stock_prices.csv
│   └── model_dataset.csv
└── outputs/
    ├── model_comparison.csv
    ├── model_accuracy_comparison.png
    ├── feature_importance.csv
    ├── logistic_confusion_matrix.csv
    ├── random_forest_confusion_matrix.csv
    ├── xgboost_confusion_matrix.csv
    └── baseline_confusion_matrix.csv
```

This structure describes the intended project organization. The `data/` files are generated by the preparation script and need not be committed if you prefer to keep the repository limited to code and evaluation outputs.

## How to Run

### 1. Install R

Install R and, optionally, RStudio.

### 2. Install the Required Packages

Run the following command in R:

```r
install.packages(c(
  "quantmod",
  "TTR",
  "dplyr",
  "ggplot2",
  "ranger",
  "xgboost",
  "pROC"
))
```

### 3. Clone or Download the Repository

Download the repository from GitHub and open its root folder in RStudio.

### 4. Prepare the Dataset

Run the data-preparation script first:

```r
source("data_preparation.R")
```

This script downloads historical stock data and creates:

- `data/stock_prices.csv`
- `data/model_dataset.csv`

An internet connection and access to Yahoo Finance are required.

### 5. Train and Evaluate the Models

After data preparation finishes successfully, run:

```r
source("stock_prediction.R")
```

The modelling script trains the three classifiers, calculates evaluation metrics, and saves the results and charts in the `outputs/` directory.

Run both scripts from the project root so the relative file paths resolve correctly.

## Limitations

- The experiment focuses on a single stock.
- Predictive performance is only modestly above the baseline for the best-performing model.
- Technical indicators may not capture all information influencing stock prices.
- The results depend on the selected period, features, model settings, and classification threshold.
- This experiment does not establish profitability after transaction costs, slippage, taxes, or other trading expenses.
- The results have not established that the models generalize to other assets or future market regimes.
- The latest experiment uses a single chronological holdout rather than a full walk-forward evaluation.

## Future Work

- Extend the analysis to multiple stocks and asset classes.
- Evaluate walk-forward validation and multiple market regimes.
- Investigate feature selection and additional market indicators.
- Tune model hyperparameters using time-aware validation.
- Add transaction-cost-aware backtesting and portfolio-level risk metrics.
- Explore probability calibration and alternative classification thresholds.
- Investigate computational efficiency and implementation considerations relevant to quantitative development.

## Disclaimer

This project is for educational and research purposes only. It is not investment advice and does not guarantee future market performance or trading profits.
