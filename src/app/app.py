import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st


start = '2000-01-01'
end = '2022-01-16'


st.title('Stock Recomendation System.')

user_input = st.text_input('Enter Stock Ticker' , 'ICICIBANK.NS')
df = yf.download(user_input , start , end)

#Describing Data
st.subheader('Data from 2010 - 2022')
st.write(df.describe())

#visualizations

st.subheader('Closing price Vs Time chart')

fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing price Vs Time chart with Moving average.')

fig = plt.figure(figsize=(12,6))

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)


















