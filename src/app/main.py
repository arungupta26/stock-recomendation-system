# pip install streamlit fbprophet yfinance plotly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st

from datetime import date

from plotly import graph_objs as go


### all functions
@st.cache
def load_data(ticker):
	data = yf.download(ticker , START , TODAY)
	data.reset_index(inplace=True)
	return load_data



# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)




### all functions end





START = "2015-01-01"

TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Recomendation System")

st.subheader('Sem-IV, BITS PIlani, WILP.  Arun Gupta(2019AP04010)')



tickers = pd.read_html('https://ournifty.com/stock-list-in-nse-fo-futures-and-options.html#:~:text=NSE%20F%26O%20Stock%20List%3A%20%20%20%20SL,%20%201000%20%2052%20more%20rows%20')[0]

tickers = tickers.SYMBOL.to_list()

for count in range(len(tickers)):
  tickers[count] = tickers[count] + ".NS"



selected_stock = st.selectbox("Select the stock." , tuple(tickers))

n_years = st.slider("Years of prediction",1,4)

period = n_years * 365


data_load_state = st.text("Load data...")
data= load_data(selected_stock)
data_load_state.text("Loading data... Done.")


st.subheader("Raw data")


plot_raw_data()

# # Predict forecast with Prophet.
# df_train = data[['Date','Close']]
# df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=period)
# forecast = m.predict(future)

# # Show and plot forecast
# st.subheader('Forecast data')
# st.write(forecast.tail())


m = load_model('keras_model_lstm.h5')

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)





















