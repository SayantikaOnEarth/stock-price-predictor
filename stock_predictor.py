import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Download stock data
ticker = 'TCS.NS'  # You can also try 'AAPL', 'INFY.NS'
data = yf.download(ticker, start="2015-01-01", end="2024-12-31")

# Prepare data
data = data[['Close']]
data['Tomorrow'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Split data
X = np.array(data[['Close']])
y = np.array(data['Tomorrow'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title(f'{ticker} - Actual vs Predicted Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
