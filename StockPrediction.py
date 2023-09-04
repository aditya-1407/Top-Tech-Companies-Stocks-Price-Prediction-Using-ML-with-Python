import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Define the stock information
stockName = "AAPL"
# stockName = "MSFT"
# stockName = "GOOG"
# stockName = "META"
startDate = "2023-01-01"
endDate = "2023-07-31"

# Download stock data from Yahoo Finance
stockRecord = yf.download(stockName, start=startDate, end=endDate)

# Calculate daily returns
stockRecord['DailyReturn'] = stockRecord['Adj Close'].pct_change()

# Drop rows with NaN values
stockRecord = stockRecord.dropna()

# Define target features and the target variable
targetHeading = ['Open', 'High', 'Low', 'Volume', "Close"]
x = stockRecord[targetHeading]
y = stockRecord['DailyReturn']

# Standardize the feature data
scaler = StandardScaler()
xScaler = scaler.fit_transform(x)

# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(xScaler, y, test_size=0.3, random_state=42)

# Create a linear regression model
lr = LinearRegression()
lr.fit(xTrain, yTrain)

# Make predictions
yPredict = lr.predict(xTest)

# Calculate Mean Squared Error
mse = mean_squared_error(yTest, yPredict)

# Create a DataFrame to compare actual and predicted returns
df = pd.DataFrame({'Actual Return': yTest, 'Predicted Return': yPredict})

# Plotting the results using Seaborn
sns.set(style='darkgrid')
sns.lineplot(data=df, markers=True, errorbar=None)
plt.title(f"{stockName} Stock Price Prediction Using Linear Regression", fontname="Times New Roman", fontsize=20)
plt.xlabel("Month")
plt.ylabel("Price")
plt.show()

# Print the Mean Squared Error
print(f"Mean Squared Error: {mse:.4f}")
