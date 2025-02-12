import streamlit as st
import pandas as pd
import requests
import datetime
from prophet import Prophet
import time
import matplotlib.pyplot as plt

# Streamlit UI settings
st.set_page_config(page_title="Crypto Price Prediction", page_icon=":chart_with_upwards_trend:")

# Title
st.title("Crypto Price Prediction with AI Model")
st.markdown("**Retrieve historical data from Gate.io and predict future prices using the Prophet model.**")   



# Display Prophet Introduction
st.markdown("""
#### 📌 What is Prophet?
Prophet is a **time series forecasting model** developed by Facebook (now Meta). It is designed to handle **trend and seasonality** automatically, making it ideal for business and financial forecasting. Prophet is based on an **additive model**, allowing it to capture trends, seasonal patterns, and external influences effectively.

#### 📊 Key Features of Prophet
- **Automatic Trend Detection** – Supports both linear and non-linear trends with changepoint detection.  
- **Multi-Seasonality Support** – Handles **daily, weekly, and yearly seasonality** without manual configuration.  
- **Robust to Missing Data & Outliers** – Can handle missing values and is less sensitive to extreme fluctuations.  
- **External Regressors Support** – Allows integration of external factors like **holidays, market events, and custom variables** to enhance accuracy.  
- **Irregular Time Interval Handling** – Unlike ARIMA, Prophet can work with **irregularly spaced data points**, making it more flexible.  
""")


# User input parameters
# symbol = st.text_input("Enter Trading Pair (e.g., BTC_USDT):", "BTC_USDT")
# interval = st.selectbox("Select Data Frequency:", ["4h", "8h", "1d"], index=2)
# history_points = st.slider("Number of historical data points: (Pls Check the number of historical data points before enter)", min_value=50, max_value=3000, value=300)
# future_points = st.slider("Number of future data points to predict:", min_value=10, max_value=1000, value=100)
symbol = st.text_input(
    "Enter Trading Pair (e.g., BTC_USDT):", 
    "BTC_USDT",
    help="Enter the trading pair in uppercase format, e.g., BTC_USDT."
)

interval = st.selectbox(
    "Select Data Frequency:",
    ["4h", "8h", "1d"],
    index=2,
    help="Choose the time interval for historical data. Smaller intervals provide more granular data."
)

history_points = st.slider(
    "Number of historical data points:", 
    min_value=50, 
    max_value=3000, 
    value=300,
    help="Select the number of past data points to fetch. Ensure the token has sufficient historical data."
)

future_points = st.slider(
    "Number of future data points to predict:", 
    min_value=10, 
    max_value=1000, 
    value=100,
    help="Select how many future data points you want to predict using the model."
)


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
                🔴 Please **Ensure the token has sufficient historical data.**   
                🔴 Or **Check if the symbol is correct.**   
                🔴 Or **Try again later**.  
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


# 假设 forecast 和 df 是 Prophet 预测结果和历史数据
def plot_prediction(df, forecast, symbol):
    fig, ax = plt.subplots(figsize=(14, 7))

    # 历史数据
    ax.plot(df["ds"], df["y"], label="Historical Price", linewidth=2, color="#0068FF")
    
    # 预测数据
    ax.plot(forecast["ds"], forecast["yhat"], label="Predicted Price", linestyle="-", color="#17E6A1", linewidth=2)
    
    # 信心区间
    ax.fill_between(forecast["ds"], 
                    forecast["yhat_lower"] * 0.95, 
                    forecast["yhat_upper"] * 1.05, 
                    color="#17E6A1", alpha=0.2, label="Confidence Interval")

    # 设置 Y 轴格式
    y_ticks = ax.get_yticks()
    max_y = max(y_ticks)
    
    if max_y >= 1000:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'${int(tick):,}' for tick in y_ticks])  # 千分位格式
    elif max_y >= 1:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'${tick:.2f}' for tick in y_ticks])  # 2 位小数
    elif max_y >= 0.0000001:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'${tick:.8f}' for tick in y_ticks])  # 8 位小数
    else:
        ax.set_ylim(bottom=0)  # 确保 Y 轴最小值为 0

    # 标题 & 图例
    ax.set_title(f"{symbol} Price Prediction With Confidence Interval")
    ax.legend()
    
    # 在 Streamlit 显示 Matplotlib 图
    st.pyplot(fig)

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

        # 示例调用（请确保 df 和 forecast 已经被 Prophet 计算出来）
        st.write(f"### {symbol} Price Prediction")
        plot_prediction(df, forecast, symbol)
