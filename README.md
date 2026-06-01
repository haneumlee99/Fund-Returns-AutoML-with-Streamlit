# Fund-Returns-AutoML-with-Streamlit

This project is a Streamlit-based AutoML application that predicts whether the next period's market return will be positive or negative based on historical return data.

The app is designed for a data visualization course assignment, with emphasis on human-centered design, simple user flow, clear visual output, and model comparison.


## Overview

The app allows a user to upload periodic financial return data, automatically generate return-based features, run multiple classification models, and compare model performance through interactive visualizations.

The goal is not to create a trading system, but to provide a clear and interpretable model comparison workflow for financial time-series classification.


## Features

- CSV upload for financial return or price data
- Date and value column selection
- Automatic conversion from price to return when needed
- Missing value and summary statistics preview
- Return-based feature engineering
- Chronological train/test split for time-series data
- Four scikit-learn classification models:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - MLP Classifier
- Interactive model leaderboard
- Confusion matrix comparison
- ROC curve comparison
- Feature importance for tree-based models
- Selected model summary
- Downloadable reproducible model template


## Input Data Format

The app expects a CSV file with at least two relevant columns:

- A date column
- A return or price column

The app contains a sample data of S&P 500 2025 returns from www..investing.com/indices/us-spx-500-historical-data.
