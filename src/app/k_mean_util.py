import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

stock_basic_details = './../resources/stock_basics.csv'


def gen_clusters(features, columns, clusterNum, clusterPrint=0):
    dataset = pd.read_csv(stock_basic_details)
    stocks = dataset.iloc[:, 0].values

    kmeans = KMeans(n_clusters=clusterNum, init='k-means++', max_iter=300, n_init=10, random_state=0)

    clusterIds = kmeans.fit_predict(features[columns])

    # First initialize each cluster
    clusters = []
    for i in range(0, clusterNum):
        clusters.append([])

    # Second fill cluster with stocks
    stockId = 0
    for clusterId in clusterIds:
        stock = stocks[stockId]
        clusters[clusterId].append(stock)
        stockId += 1

        # Print out cluster
    if clusterPrint == 1:
        print("Here are generated clusters:\n")
        clusterId = 1
        for i in range(0, clusterNum):
            print("cluster-" + str(clusterId) + ": " + ",".join(clusters[i]))
            clusterId += 1
    return clusters


def in_cluster_stocks(stock_input, features, fids_concerned, level):
    # First load the dataset on stock basics
    dataset = pd.read_csv(stock_basic_details)
    stocks = dataset.iloc[:, 0].values
    # print("Below is stock list in dataset:")
    # print(stocks)
    # print("-------------------------------------------------------------")

    features = dataset.iloc[:, 1:].values
    features = pd.DataFrame(features)
    features.columns = ["Price", "Volume", "Market Cap", "Beta", "PE Ratio", "EPS"]
    cols = features.columns
    features[cols] = features[cols].apply(pd.to_numeric, errors='coerce')
    # print("Below is the data of stock features: ")
    # print(features)

    # Second we eliminate null values in the dataset
    for i in features.columns:
        features[i] = features[i].fillna(float(features[i].mean()))

    clusterNum = level * 30
    clusterPrint = 0
    columns = []
    for fid in fids_concerned:
        columns.append(features.columns[fid - 1])
    clusters = gen_clusters(features, columns, clusterNum, clusterPrint)

    # print("Stock you have input: " + stock_input)
    # print("Features you are concerned about: " + ', '.join(columns))
    # print("---------------------------------------------------")
    count = 0
    isfound = 0
    feature_based_similar_stocks = []
    for cluster in clusters:
        if stock_input in cluster and len(cluster) > 1:
            isfound = 1
            for cluster_stock in cluster:
                if cluster_stock != stock_input:
                    count += 1
                    feature_based_similar_stocks.append(cluster_stock)
            break
    if not isfound:
        print("Sorry, we can not make any recommendation based on your input")

    return feature_based_similar_stocks
