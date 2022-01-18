import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

import linear_regression_util as lrutil


st.title('Stock Recomendation System.')
st.subheader('Sem-IV, BITS PIlani, WILP.  Arun Gupta(2019AP04010)')

stock_list_file = r'../resources/stock_list.txt'

tickers = lrutil.get_list(stock_list_file)

list = []
for count in range(len(tickers)):
    list.append(tickers[count][0])

tickers = list

# tickers = tickers.SYMBOL.to_list()

# for count in range(len(tickers)):
#   tickers[count] = tickers[count] + ".NS"



user_input = st.selectbox('Please select a stock.',tuple(tickers))

#st.text_input(Enter Stock Ticker , ICICIBANK.NS)

df = yf.download(user_input , lrutil.start , lrutil.end)

#Describing Data
st.subheader('Data from 2010 - 2022')
st.write(df.describe())

#visualizations

# st.subheader(Closing price Vs Time chart)

# fig = plt.figure(figsize=(12,6))
# plt.plot(df.Close)
# st.pyplot(fig)


st.subheader("Closing price Vs Time chart with Moving average.")

fig = plt.figure(figsize=(12,6))

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

plt.plot(df.Close,'r',label = 'Actual Closing price')
plt.plot(ma100,'g',label = 'MA 100 Closing price')
plt.plot(ma200,'b',label = 'MA 200 Closing price')
plt.xlabel('Time')
plt.ylabel('Closing Price')

plt.legend()
st.pyplot(fig)



recomended_stocks = lrutil.top_five_stock(stock_list_file)
st.subheader('The top five recomended stock as date : '+ lrutil.end + ' are ' +recomended_stocks  )




