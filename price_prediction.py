import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt
import time

# 设置 Streamlit 页面
st.set_page_config(page_title='BTC Price Prediction', page_icon=':chart_with_upwards_trend:')

st.title("BTC 价格预测 (Gate.io API + Prophet)")

# 选择交易对
symbol = st.text_input("输入交易对 (如 BTC_USDT):", "BTC_USDT")

# 选择数据频率
interval_options = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
interval = st.selectbox("选择数据频率:", list(interval_options.keys()), index=5)
interval_seconds = interval_options[interval]

# 选择数据点数
num_points = st.slider("选择要获取的历史数据点数:", min_value=100, max_value=1000, value=500, step=50)

# 选择预测点数
future_points = st.slider("选择预测的未来数据点数:", min_value=10, max_value=500, value=200, step=10)

# 获取 Gate.io 价格数据
def fetch_gateio_data(symbol, interval, num_points):
    base_url = "https://api.gateio.ws/api/v4/spot/candlesticks"
    params = {
        "currency_pair": symbol,
        "interval": interval,
        "limit": num_points
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit='s')
        df["close"] = df["close"].astype(float)
        return df
    else:
        st.error(f"获取数据失败: {response.status_code}")
        return None

# 获取数据
st.write("正在获取历史数据...")
df = fetch_gateio_data(symbol, interval, num_points)
if df is not None:
    st.write("数据加载完成!")
    st.write(df.tail())
    
    # 准备数据给 Prophet
    df_prophet = df[["timestamp", "close"]].rename(columns={"timestamp": "ds", "close": "y"})
    
    # 训练 Prophet 模型
    st.write("训练 Prophet 模型...")
    model = Prophet()
    model.fit(df_prophet)
    
    # 生成未来时间点
    future = model.make_future_dataframe(periods=future_points, freq=interval)
    
    # 进行预测
    forecast = model.predict(future)
    
    # 绘制预测结果
    st.write("预测结果:")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_prophet["ds"], df_prophet["y"], label="历史价格", linewidth=2)
    ax.plot(forecast["ds"], forecast["yhat"], label="预测价格", linestyle="--", color="orange", linewidth=2)
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color='orange', alpha=0.2, label='置信区间')
    ax.set_title("BTC 价格预测")
    ax.set_xlabel("日期")
    ax.set_ylabel("价格 (USDT)")
    ax.legend()
    st.pyplot(fig)
