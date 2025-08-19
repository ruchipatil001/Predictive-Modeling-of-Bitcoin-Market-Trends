import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title="BNB/USD Forecast", layout="wide")

st.title("BNB/USD Price Prediction with Prophet")

# Step 1: Load Local Dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\ruchi\Downloads\BNBUSDT.csv")   # <-- Your dataset
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure date format
    return df

df = load_data()

st.subheader("Historical Data (BNB/USD)")

#  Step 2: Downsample if too large
if len(df) > 500000:  # If more than 1 lakh rows, keep last 50k
    df = df.tail(200000)

#  Step 2: Prepare Data for Prophet
df_train = df[['timestamp', 'close']].rename(columns={"timestamp": "ds", "close": "y"})

#  Step 3: User Input-Forecast Period
n_days = st.slider("Forecast days:", min_value=7, max_value=180, value=30)

#  Step 4: Train Prophet Model
@st.cache_resource
def train_model(data):
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    return model

model = train_model(df_train)

#  Step 5: Make Future Predictions
future = model.make_future_dataframe(periods=n_days)
forecast = model.predict(future)

#  Step 6: Display Forecast
st.subheader(" Forecasted Prices")
fig = plot_plotly(model, forecast)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast Data (last 10 days)")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

st.success(" Prediction completed successfully!")