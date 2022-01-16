import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st



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




### all functions

def load_data(ticker):
	data = yf.download(user_input , START , TODAY)
	data.reset_index(inplace=True)
	return load_data

























