import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import fbprophet_util as fbp_util
import google_news_util as gnu
import k_mean_util as kutil
import linear_regression_util as lrutil

title_alignment = """
<style>
#the-title {
  text-align: center
}
</style>
"""
st.markdown(title_alignment, unsafe_allow_html=True)

st.title('Stock Recommendation System.')
st.subheader('Sem-IV, BITS PIlani, WILP.')
st.subheader('Name: Arun Gupta (2019AP04010)')

tickers = lrutil.get_list(lrutil.stock_list_file)

list = []
for count in range(len(tickers)):
    list.append(tickers[count][0])

tickers = list

user_input = st.selectbox('Please select a stock.', tuple(tickers))

df = yf.download(user_input, lrutil.start, lrutil.end)

st.subheader("Closing price Vs Time chart with Moving average.")

fig = plt.figure(figsize=(12, 6))

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

plt.plot(df.Close, 'r', label='Actual Closing price')
plt.plot(ma100, 'g', label='MA 100 Closing price')
plt.plot(ma200, 'b', label='MA 200 Closing price')
plt.xlabel('Time')
plt.ylabel('Closing Price')

plt.legend()
st.pyplot(fig)

c1, c2 = st.columns(2)

with c1:
    predicted_data = fbp_util.get_predicted_price(user_input)
    predicted_data = predicted_data.reset_index()

    predicted_data['ds'] = predicted_data['ds'].dt.strftime('%m/%d/%Y')
    predicted_data.rename(columns={'ds': 'Future dates'}, inplace=True)
    st.subheader("Last day closing price for " + user_input + " was Rs:" + f"{(df.iloc[-1]['Close']):.4f}")
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

    st.table(predicted_data)

with c2:
    features = ["Price", "Volume", "Market Cap", "Beta", "PE Ratio", "EPS"]

    st.subheader("Stocks you may be interested based on features selected and accuracy level")
    selected_features = st.multiselect("Please select the features", features, ["Volume", "Price"])
    level = st.select_slider('Please select accuracy level.(Being 1 as LOW and 3 as HIGH)', options=[1, 2, 3], value=2)

    selected_feature_index = []
    index = 1
    for f in features:
        if f in selected_features:
            selected_feature_index.append(index)
        index = index + 1

    if len(selected_feature_index) == 0:
        st.markdown("Please select at least one feature.")

    if len(selected_feature_index) > 0:
        feature_based_similar_stocks = kutil.in_cluster_stocks(user_input, features, selected_feature_index, level)

        if len(feature_based_similar_stocks) == 0:
            st.markdown("Sorry, we can not make any recommendation based on your input")
        if len(feature_based_similar_stocks) > 0:
            st.table(
                pd.DataFrame(feature_based_similar_stocks,
                             index=(i + 1 for i in range(len(feature_based_similar_stocks))),
                             columns=['Based on Features and accuracy level selected, Stocks are']))

top_performing_stocks = lrutil.top_five_stock(lrutil.stock_list_file, lrutil.stock_coefficient_file_name)

similar_stocks = lrutil.similar_stocks(symbol=user_input)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Today's High performing stocks (Top Five)")
    st.table(
        pd.DataFrame(top_performing_stocks['Stock'].tolist(), index=(i + 1 for i in range(len(top_performing_stocks))),
                     columns=['Name']))

with col2:
    st.subheader("Stocks similar to selected : " + user_input)
    st.table(pd.DataFrame(similar_stocks['Stock'].tolist(), index=(i + 1 for i in range(len(similar_stocks))),
                          columns=["Name"]))

with col3:
    posDelta, negDelta = lrutil.dailyTop5Delta()

    # l1, l2 = st.columns(2)
    #
    # with l1:
    st.subheader("Top gainers of the day(in %)")
    st.table(posDelta)

with col4:
    st.subheader("Top losers of the day(in %)")
    st.table(negDelta)

allnewstitle, allnewssummary, positive, neutral, negative = gnu.fetch_google_sentimental_analysis(user_input)

total_news = (positive + neutral + negative)

n1, n2 = st.columns(2)

with n1:
    # st.table(allnews)
    if (total_news > 0):
        st.subheader('All news extracted from google for ' + user_input)

        index = 0
        for newstitle in allnewstitle:
            with st.expander(newstitle):
                st.write(allnewssummary[index])
    else:
        st.subheader("No news available for selected stock " + user_input)

with n2:
    if (total_news > 0):
        # Creating PieCart
        st.subheader('Sentimental analysis for selected stock ' + user_input)
        st.markdown('**Based on google news data')

        labels = ' Positive [' + str(round(positive)) + '%]' + '\n Neutral [' + str(
            round(neutral)) + '%]' + '\n Negative [' + str(round(negative)) + '%]'

        mylabels = [' Positive ', ' Neutral ', ' Negative ']

        sentimental_data = [['Positive', str(round(positive * 100 / total_news)) + '%'],
                            ['Neutral', str(round(neutral * 100 / total_news)) + '%'],
                            ['Negative', str(round(negative * 100 / total_news)) + '%']]

        sizes = [positive, neutral, negative]
        sentimental_analysis = pd.DataFrame(sentimental_data, index=(i + 1 for i in range(len(sentimental_data))),
                                            columns=['Sentiment', 'Percentage'])
        # st.table(sentimental_analysis)

        fig = plt.figure(figsize=(4, 2))
        plt.pie(sizes, labels=mylabels)

        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        mylabels = ['Positive', 'Neutral', 'Negative']
        sizes = [positive, neutral, negative]
        colors = ['green', 'yellow', 'red']

        patches, texts = plt.pie(sizes, colors=colors, shadow=False, startangle=90)
        plt.legend(patches, mylabels, loc="upper left", prop={"size": 4})
        plt.axis('equal')

        # st.pyplot(fig)

        st.bar_chart(pd.DataFrame(np.array([[positive, neutral, negative]]), columns=mylabels))

