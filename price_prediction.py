import streamlit as st
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt
from prophet import Prophet

# Streamlit UI settings
st.set_page_config(page_title="Crypto Price Prediction", page_icon=":chart_with_upwards_trend:")

# Title
st.title("Crypto Price Prediction with Prophet")
st.write("Retrieve historical data from Gate.io and predict future prices using the Prophet model.")

# User input parameters
symbol = st.text_input("Enter Trading Pair (e.g., BTC_USDT):", "BTC_USDT")
interval = st.selectbox("Select Data Frequency:", ["1m", "5m", "1h", "4h", "1d"], index=4)
history_points = st.slider("Number of historical data points:", min_value=50, max_value=500, value=200)
future_points = st.slider("Number of future data points to predict:", min_value=10, max_value=300, value=100)

# Fetch historical data from Gate.io API
def fetch_data(symbol, interval, limit):
    url = f"https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to retrieve data. Please check the trading pair or try again later.")
        return None

# Process data
def process_data(data):
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='s')
    df = df.sort_values(by="timestamp")
    df.rename(columns={"timestamp": "ds", "close": "y"}, inplace=True)
    df["y"] = df["y"].astype(float)
    return df

# Run data fetching and prediction
if st.button("Run Prediction"):
    st.write("Fetching data...")
    raw_data = fetch_data(symbol, interval, history_points)
    if raw_data:
        df = process_data(raw_data)
        
        # Initialize Prophet model with tuning parameters
        model = Prophet(
            changepoint_prior_scale=0.1,  # Increase sensitivity to detect trend changes
            interval_width=0.2,  # Confidence interval
            uncertainty_samples=500,  # Reduce uncertainty
            daily_seasonality=True
        )
        
        # Add seasonality based on the interval
        if interval == "1h":
            model.add_seasonality(name='daily', period=24, fourier_order=6)
        elif interval == "4h":
            model.add_seasonality(name='daily', period=6, fourier_order=6)
        elif interval == "1d":
            model.add_seasonality(name='weekly', period=7, fourier_order=3)
        
        model.fit(df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=future_points, freq=interval)
        forecast = model.predict(future)
        
        # Plot results
        # Display the forecast using Streamlit's built-in charting function
        st.write("### Price Prediction Chart")
        
        forecast_display = forecast[['ds', 'yhat']]
        forecast_display = forecast_display.rename(columns={"ds": "Date", "yhat": "Predicted Price"})
        forecast_display.set_index("Date", inplace=True)
        
        historical_display = df[['ds', 'y']]
        historical_display = historical_display.rename(columns={"ds": "Date", "y": "Historical Price"})
        historical_display.set_index("Date", inplace=True)
        
        # Combine historical and predicted data
        combined_data = historical_display.join(forecast_display, how="outer")
        
        st.line_chart(combined_data)
        
        # # Show forecast data
        # st.write("### Forecasted Prices")
        # st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_points))
