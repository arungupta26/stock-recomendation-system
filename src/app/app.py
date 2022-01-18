import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

import linear_regression_util as lrutil


st.title('Stock Recomendation System.')
st.subheader('Sem-IV, BITS PIlani, WILP.')
st.subheader('Name: Arun Gupta (2019AP04010)')

tickers = lrutil.get_list(lrutil.stock_list_file)

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
# st.subheader('Data from 2000 - 2022')
# st.write(df.describe())

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


recomended_stocks = lrutil.top_five_stock(lrutil.stock_list_file, lrutil.stock_coefficient_file_name)


similar_stocks = lrutil.similar_stocks(symbol=user_input)


col1, col2 = st.columns(2)

with col1:
    st.markdown("Top 5 recommended stocks")
    st.table( pd.DataFrame(recomended_stocks['Stock'].tolist(),index=( i+1 for i in range(len(recomended_stocks))),columns=['Name']))

with col2:
    st.markdown("Stocks similar to selected : "+user_input)
    st.table(pd.DataFrame(similar_stocks['Stock'].tolist(),index=( i+1 for i in range(len(similar_stocks))),columns=["Name"]))






