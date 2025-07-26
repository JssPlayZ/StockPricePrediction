import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# App Title
st.title("📈 Stock Price Prediction with LSTM")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Load Data
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    df['MA14'] = df['Close'].rolling(window=14).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    return df

df = load_data(ticker, start_date, end_date)
st.subheader(f"{ticker} - Stock Closing Price")
st.line_chart(df['Close'])

# Normalize and sequence data
features = ['Close', 'MA14', 'RSI']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])  # Close
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Load trained model
try:
    model = load_model("lstm_model.h5")
except Exception as e:
    st.error("❌ Trained model file 'lstm_model.h5' not found.")
    st.stop()

# Predict
y_pred_scaled = model.predict(X_test)

# Inverse transform predictions
def inverse_transform_column(column):
    padded = np.concatenate([column, np.zeros((len(column), 2))], axis=1)
    return scaler.inverse_transform(padded)[:, 0]

y_test_actual = inverse_transform_column(y_test.reshape(-1, 1))
y_pred_actual = inverse_transform_column(y_pred_scaled)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

# Show metrics
st.subheader("📊 Model Evaluation Metrics")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Plot Actual vs Predicted
st.subheader("📉 Actual vs Predicted Stock Price")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test_actual, label='Actual Price')
ax.plot(y_pred_actual, label='Predicted Price')
ax.set_title('Actual vs Predicted')
ax.set_xlabel('Time Step')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True)
st.pyplot(fig)
