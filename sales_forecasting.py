import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv("sales_data.csv")

data['Date'] = pd.to_datetime(data['Date'])

data['Month'] = np.arange(len(data))

X = data[['Month']]
y = data['Sales']

model = LinearRegression()
model.fit(X, y)

future_months = np.arange(len(data), len(data)+6).reshape(-1,1)

predictions = model.predict(future_months)

plt.plot(data['Date'], y, label="Actual Sales")

future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=7, freq='M')[1:]

plt.plot(future_dates, predictions, label="Forecast Sales")

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecast")

plt.legend()
plt.show()
