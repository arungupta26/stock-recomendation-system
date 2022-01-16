import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st


start = '2000-01-01'
end = '2022-01-11'


st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker' , 'ICICIBANK.NS')
df = yf.download(user_input , start , end)

#Describing Data
st.subheader('Data from 2010 - 2022')
st.write(df.describe())