import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
import matplotlib.pyplot as pl
import yfinance as yf
from datetime import date
import os.path
import streamlit as st

from pandas_datareader._utils import RemoteDataError

from sklearn.model_selection import train_test_split

stock_list_file = r'../resources/stock_list.txt'
stock_coefficient_file_name = './../resources/Stock_Coefficients.csv'

# Considering data from 2000 to till date for latest trend
start = '2018-01-01'
end = date.today().strftime("%Y-%m-%d")
#end = '2022-02-05'


# function1 to get stock list from text file
def get_list(filename):
    # pandas read file into data frame
    stocks = pd.read_csv(filename, header=None)
    stocks[0] = stocks[0].str.upper()
    # convert the data frame to numpy array
    stock_array = stocks.values
    return stock_array


# function2: get historical data for a specific stock, using symbol as the parameter
def load_stock_data(symbol):
    try:
        df = yf.download(symbol, start, end)
        # web.DataReader(symbol, 'yahoo', start, end)
    except RemoteDataError:
        print("No search result : '" + symbol + "'")
        return float('NaN')
    except KeyError:
        print("Date range not supported : '" + symbol + "'")
        return float('NaN')
    close = df[['Close']]

    base = close.iloc[0]['Close']
    if base > 10:
        close = close.assign(Close=close['Close'] / base * 100)
    return close


# function3: build linear regression model for a specific stock
# parameter
# show_statistics: indicate if the statistics are printed
# show_dots: indicate if dots data needed
# show_plot: indacate if the plot is shown

# The coefficient estimates for Ordinary Least Squares rely on the independence of the features.
# When features are correlated and the columns of the design matrix  have an approximately linear dependence,
# the design matrix becomes close to singular and as a result, the least-squares estimate becomes highly sensitive to random errors in the observed target,
# producing a large variance.

def build_linear_regression(symbol, show_statistics=True, show_dots=True, show_plot=False):
    close = load_stock_data(symbol)
    # if the close is not dataframe, return NaN
    if not isinstance(close, pd.DataFrame):
        return float('NaN')

    # normalize datatime datatype to integer
    # simply converting datetime's to # of days since 2018-01-01 divided by 100
    close.index = (close.index - pd.to_datetime('2018-01-01')).days / 100
    close = close.reset_index()
    train, test = train_test_split(close)

    train_x = train.drop('Close', axis=1)
    train_y = train[['Close']]
    test_x = test.drop('Close', axis=1)
    test_y = test[['Close']]

    # call linear regression model
    regr = linear_model.LinearRegression()
    regr.fit(train_x, train_y)

    if show_statistics:
        # The coefficients
        print('Coefficients:', regr.coef_[0])
        # The mean squared error
        print("Mean squared error: %.2f"
              % np.mean((regr.predict(test_x) - test_y) ** 2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % regr.score(test_x, test_y))
        print('features:', 'Date', 'Close_Price')

    if show_plot:
        pl.plot(test_x, regr.predict(test_x), label=symbol)
        pl.legend()

        # Plot outputs
        pl.xlabel('Date')
        pl.ylabel('Close Price')
        pl.title('Overall Linear Regression Model')

        if show_dots:
            # pl.title(symbol + ' Linear Regression Model')
            pl.plot(test['Date'], test['Close'], linestyle='none', marker='o')

        pl.xticks(())
        pl.yticks(())
    # return the coefficient representing trend
    return regr.coef_[0][0]


# function4: for each symbol in the text file, caculate the coefficents and record in the original text file
def get_coefficient_dataset(filename, show_statistics=False, show_dots=False, show_plot=False):
    # load stock list data from text file
    stock_array = get_list(filename)
    # extend the 2D array from N * 1 to N * 2 to make place for putting corresponding coefficient
    stock_array = np.insert(stock_array, 1, values=0, axis=1)

    # caculate coefficient for each symbol and store in the stock_array
    for symbol in stock_array:
        print(symbol)
        coefficient = build_linear_regression(symbol[0], show_statistics, show_dots, show_plot)
        symbol[1] = coefficient

    # transfer stock_array to dataframe with two columns 'Stock' and 'Coefficient'
    coefficient_data = pd.DataFrame(stock_array)
    coefficient_data.columns = ['Stock', 'Coefficient']

    # store the dataframe to csv file for future using
    coefficient_data.to_csv(stock_coefficient_file_name)
    return coefficient_data


# function5 to get top n stocks with biggest coefficients
def get_top_stock(coefficient_data, n=5, show_dots=False):
    # sort stocks by coefficients with descending order
    top_stocks = coefficient_data.sort_values(by=['Coefficient'], ascending=False)
    # take top n stocks
    top_stocks = top_stocks[:n]
    top_stocks_list = top_stocks['Stock'].tolist()
    # plot stocks
    for symbol in top_stocks_list:
        build_linear_regression(symbol, show_statistics=False, show_dots=show_dots, show_plot=True)
    # print stock list
    print("The " + str(n) + " stocks with best trends are: ")
    print(top_stocks['Stock'].tolist())
    return top_stocks


@st.cache
def top_five_stock(stock_list_file, stock_coefficient_file_name):
    file_exists = os.path.exists(stock_coefficient_file_name)
    if not file_exists:
        get_coefficient_dataset(stock_list_file, show_plot=False)

    coefficient_data = pd.read_csv(stock_coefficient_file_name)
    top_performing_stocks = get_top_stock(coefficient_data, n=5)
    pd.DataFrame(top_performing_stocks, index=(i + 1 for i in range(len(top_performing_stocks))), columns=['Name'])
    return top_performing_stocks


# Function6 to get stocks with same trends
def get_similar_stock(coefficient_data, symbol, n=3, show_dots=False):
    # sort stocks by coefficients with descending order
    top_stocks = coefficient_data.sort_values(by=['Coefficient'], ascending=False)
    # get the specific row of the stcok symbol
    row = top_stocks.loc[top_stocks['Stock'] == symbol].iloc[-1]
    # get the offset in the column of that row in the sorted table
    offset = top_stocks.index.get_loc(row.name)

    # take similar stocks in the table from offset-3 to offset+3
    start_offset = offset - n
    end_offset = offset + n

    if start_offset < 0:
        start_offset = 0

    similar_stocks = top_stocks[start_offset: end_offset]
    similar_stocks_list = similar_stocks['Stock'].tolist()
    # plot stocks
    for symbol in similar_stocks_list:
        print("Linear regression for stock ", symbol)
        build_linear_regression(symbol, show_statistics=False, show_dots=show_dots, show_plot=True)

    # remove the row for the input symbol
    print(row)

    similar_stocks_delete = similar_stocks.drop(row.name)
    number = n * 2 - 1
    print("The " + str(number) + " simialr stocks are: ")
    print(similar_stocks_delete['Stock'].tolist())
    return similar_stocks


def similar_stocks(symbol):
    file_exists = os.path.exists(stock_coefficient_file_name)
    if not file_exists:
        get_coefficient_dataset(stock_list_file, show_plot=False)

    coefficient_data = pd.read_csv(stock_coefficient_file_name)
    return get_similar_stock(coefficient_data, symbol)

@st.cache
def dailyTop5Delta():
    delta_df = pd.read_csv(stock_coefficient_file_name)
    delta_df = delta_df[['Stock', 'LastDayChange']]
    delta_df_positive = delta_df[delta_df.LastDayChange > 0]
    delta_df_positive = delta_df_positive.sort_values(by=['LastDayChange'], ascending=False)[:5]

    delta_df_negative = delta_df[delta_df.LastDayChange < 0].sort_values(by=['LastDayChange'], ascending=True)[:5]

    return delta_df_positive, delta_df_negative
