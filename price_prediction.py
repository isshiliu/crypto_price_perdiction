import streamlit as st
import pandas as pd
import requests
import datetime
from prophet import Prophet
import time

# Streamlit UI settings
st.set_page_config(page_title="Crypto Price Prediction", page_icon=":chart_with_upwards_trend:")

# Title
st.title("Crypto Price Prediction with AI Model")
st.write("Retrieve historical data from Gate.io and predict future prices using the Prophet model.")

# User input parameters
symbol = st.text_input("Enter Trading Pair (e.g., BTC_USDT):", "BTC_USDT")
interval = st.selectbox("Select Data Frequency:", ["4h", "8h", "1d"], index=2)
history_points = st.slider("Number of historical data points:", min_value=50, max_value=3000, value=300)
future_points = st.slider("Number of future data points to predict:", min_value=10, max_value=1000, value=100)

BASE_URL = "https://api.gateio.ws/api/v4"
LIMIT_PER_REQUEST = 100  # API限制每次最多获取100条数据


# Fetch historical data from Gate.io API
def fetch_data(symbol, interval, total_records):
    """
    Fetch historical price data from Gate.io API.
    If history_points exceed the earliest available date, adjust the range.
    """
    all_data = []
    to_time = int(time.time())  # Current timestamp

    while len(all_data) < total_records:
        from_time = to_time - LIMIT_PER_REQUEST * 86400  # 每次往前取100天
        url = f"{BASE_URL}/spot/candlesticks"
        params = {
            "currency_pair": symbol,
            "interval": interval,
            "limit": LIMIT_PER_REQUEST,
            "from": from_time,
            "to": to_time
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()

            if not data:
                st.markdown(f"""
                ❌ **ERROR No data available for `{symbol}`**  
                🔹 Please check if the symbol is correct.  
                🔹 Or change the number of historical data points.  
                🔹 Or try again later.
                """)
                # st.error(f"No data available for {symbol}./n Please check if the symbol is correct /n or change Number of historical data points /n or Try Again later.")
                return pd.DataFrame()  # Return empty DataFrame

            all_data.extend(data)
            to_time = from_time  # Move backward in time

        elif response.status_code == 400:
            st.error(f"Error Code 400, Invalid symbol: {symbol}. Please check and enter a valid trading pair.")
            return pd.DataFrame()
        
        else:
            st.error(f"Request failed with status code: {response.status_code}")
            return pd.DataFrame()

    # if not all_data:
    #     st.error(f"Failed to retrieve data for {symbol}. Please check if the symbol exists.")
    #     return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=["timestamp", "volume_token", "high", "low", "close", "open", "quote_volume", "tag"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='s')
    df = df.sort_values(by="timestamp")
    df.rename(columns={"timestamp": "ds", "close": "y"}, inplace=True)
    df["y"] = df["y"].astype(float)
    return df

# Run data fetching and prediction
if st.button("Run Prediction"):
    st.write("Fetching data...")
    df = fetch_data(symbol, interval, history_points)
    if not df.empty:
        
        # Initialize Prophet model with tuning parameters
        model = Prophet(
            changepoint_prior_scale=0.1,  # Increase sensitivity to detect trend changes
            interval_width=0.2,  # Confidence interval
            uncertainty_samples=500,  # Reduce uncertainty
            daily_seasonality=True
        )
        
        # Add seasonality based on the interval
        if interval in ["1h", "4h"]:
            model.add_seasonality(name='daily', period=24, fourier_order=6)
        elif interval == "1d":
            model.add_seasonality(name='weekly', period=7, fourier_order=3)
        
        model.fit(df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=future_points, freq=interval)
        forecast = model.predict(future)
        
        # # Display results
        # st.write("### Forecasted Prices")
        # st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_points))
        
        # Plot results using Streamlit's built-in line chart
        st.write(f"### {symbol} Price Prediction")
        forecast_display = forecast[['ds', 'yhat']]
        forecast_display = forecast_display.rename(columns={"ds": "Date", "yhat": "Predicted Price"})
        forecast_display.set_index("Date", inplace=True)
        
        historical_display = df[['ds', 'y']]
        historical_display = historical_display.rename(columns={"ds": "Date", "y": "Historical Price"})
        historical_display.set_index("Date", inplace=True)
        
        combined_data = historical_display.join(forecast_display, how="outer")
        st.line_chart(combined_data, color=["#0068FF", "#17E6A1"])
