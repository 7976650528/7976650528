pip install requests pandas numpy matplotlib tensorflow flask
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Binance API endpoint for historical data
url = 'https://api.binance.com/api/v3/klines'

# Function to fetch data from Binance API (1-minute interval)
def fetch_data(symbol='BTCUSDT', interval='1m', limit=500):
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]

# Fetching the last 500 data points
df = fetch_data()

# Convert 'close' column to float for model use
df['close'] = df['close'].astype(float)

# Plotting the Bitcoin closing prices
plt.figure(figsize=(10,6))
plt.plot(df.index, df['close'], label='Bitcoin Price (USDT)', color='blue')
plt.title('Bitcoin Price (1-Minute Interval)')
plt.xlabel('Time')
plt.ylabel('Price (USDT)')
plt.legend()
plt.show()
# Scaling the data to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['close']])

# Function to create dataset for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Create the dataset
time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape X to 3D for LSTM [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))  # Output layer for price prediction

model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X, y, epochs=10, batch_size=32)def predict_next_price(model, data):
    # Scaling the input data
    data = scaler.transform(data)
    data = data.reshape(1, 60, 1)  # Reshaping for LSTM model
    prediction = model.predict(data)
    predicted_price = scaler.inverse_transform(prediction)
    return predicted_price[0][0]

# Example of real-time prediction using the last 60 data points
latest_data = df[['close']].tail(60).values  # Last 60 closing prices
predicted_price = predict_next_price(model, latest_data)
print(f"Predicted Next Price: {predicted_price} USDT")
pip install flask
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def get_prediction():
    # Get the latest 60 data points for prediction
    latest_data = df[['close']].tail(60).values
    predicted_price = predict_next_price(model, latest_data)
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
