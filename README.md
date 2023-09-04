# Stock Price Prediction using Linear Regression

This Python script demonstrates how to perform stock price prediction using linear regression. It utilizes historical stock data obtained from Yahoo Finance, preprocesses the data, builds a linear regression model, evaluates the model's performance, and visualizes the results using Seaborn and Matplotlib.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Usage](#usage)
4. [Code Explanation](#code-explanation)
5. [Results](#results)
6. [Conclusion](#conclusion)

## Introduction

Stock price prediction is a common application of machine learning, particularly using regression models. In this script, we use a linear regression model to predict daily returns of a specified stock, in this case, Apple Inc. (AAPL). However, you can easily adapt the code to work with other stocks by changing the `stockName` variable.

## Prerequisites

Before running the script, ensure you have the following dependencies installed:

- Python (>=3.6)
- Pandas
- yfinance
- Matplotlib
- Seaborn
- scikit-learn

You can install these packages using `pip`:

```
pip install pandas yfinance matplotlib seaborn scikit-learn
```

## Usage

1. Open a terminal or command prompt.
2. Navigate to the directory where the script is located.
3. Run the script using the following command:

```
python stock_price_prediction.py
```

## Code Explanation

The code is structured as follows:

1. **Data Collection**: We use the `yfinance` library to download historical stock data for the specified stock (`stockName`) and date range (`startDate` to `endDate`).

2. **Data Preprocessing**: We calculate the daily returns as the percentage change in the adjusted closing price and remove rows with NaN values.

3. **Feature Selection**: We select target features (`targetHeading`) and the target variable (`y`) for prediction.

4. **Data Standardization**: The feature data is standardized using `StandardScaler` to have zero mean and unit variance.

5. **Data Splitting**: We split the data into training and testing sets for model evaluation.

6. **Model Training**: We create a linear regression model (`LinearRegression`) and train it on the training data.

7. **Prediction and Evaluation**: We make predictions on the testing data and calculate the Mean Squared Error (MSE) to assess the model's accuracy.

8. **Visualization**: We use Seaborn and Matplotlib to create a line plot comparing actual and predicted returns.

9. **Print Results**: The script prints the Mean Squared Error as a measure of prediction performance.

## Results

The script generates a line plot showing the actual and predicted returns for the specified stock. You can visually inspect how well the linear regression model fits the data. Additionally, the Mean Squared Error is printed, which quantifies the model's prediction accuracy.

## Conclusion

This Python script provides a practical example of stock price prediction using a linear regression model. You can modify the script to work with different stocks or customize it for more advanced analyses.
