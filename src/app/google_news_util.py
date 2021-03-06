import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
from wordcloud import WordCloud, STOPWORDS

nltk.download('vader_lexicon')


def percentage(part, whole):
    return 100 * float(part) / float(whole)


def fetch_google_sentimental_analysis(stock_name, past_days=5):
    now = dt.date.today()
    now = now.strftime('%m-%d-%Y')
    yesterday = dt.date.today() - dt.timedelta(days=past_days)
    yesterday = yesterday.strftime('%m-%d-%Y')

    nltk.download('punkt')
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10

    # Extract News with Google News
    googlenews = GoogleNews(start=yesterday, end=now)
    googlenews.search(stock_name)
    result = googlenews.result()

    # store the results
    df = pd.DataFrame(result)

    news_df = pd.DataFrame()

    # Assigning Initial Values
    positive = 0
    negative = 0
    neutral = 0
    news_summary_list = []
    news_title_list = []

    try:
        list = []  # creating an empty list
        for i in df.index:
            dict = {}  # creating an empty dictionary to append an article in every single iteration
            article = Article(df['link'][i], config=config)  # providing the link
            try:
                article.download()  # downloading the article
                article.parse()  # parsing the article
                article.nlp()  # performing natural language processing (nlp)
            except:
                pass
                # storing results in our empty dictionary
            dict['Date'] = df['date'][i]
            dict['Media'] = df['media'][i]
            dict['Title'] = article.title
            dict['Article'] = article.text
            dict['Summary'] = article.summary
            dict['Key_words'] = article.keywords
            list.append(dict)
            news_title_list.append(article.title)
            news_summary_list.append(article.summary)
        check_empty = not any(list)
        # print(check_empty)
        if not check_empty:
            news_df = pd.DataFrame(list)
        else:
            print("No news availabe in last 5 days for stock " + stock_name)
            return news_title_list, news_summary_list, positive, neutral, negative

    except Exception as e:
        # exception handling
        print("exception occurred:" + str(e))
        print(
            'Looks like, there is some error in retrieving the data, Please try again or try with a different ticker.')

    # Creating empty lists
    neutral_list = []
    negative_list = []
    positive_list = []

    if (len(news_df) > 0):
        # Iterating over the tweets in the dataframe
        for news in news_df['Summary']:
            #news_list.append(news)
            analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
            neg = analyzer['neg']
            pos = analyzer['pos']
            print(news)
            print("------------------------------------")
            print('Positive : ' + str(pos), "Negative : " + str(neg))
            print("------------------------------------")
            print("------------------------------------")

            if neg > pos:
                negative_list.append(news)  # appending the news that satisfies this condition
                negative += 1  # increasing the count by 1
            elif pos > neg:
                positive_list.append(news)  # appending the news that satisfies this condition
                positive += 1  # increasing the count by 1
            elif pos == neg:
                neutral_list.append(news)  # appending the news that satisfies this condition
                neutral += 1  # increasing the count by 1

        positive = percentage(positive, len(news_df))  # percentage is the function defined above
        negative = percentage(negative, len(news_df))
        neutral = percentage(neutral, len(news_df))

        # Converting lists to pandas dataframe
        neutral_list = pd.DataFrame(neutral_list)
        negative_list = pd.DataFrame(negative_list)
        positive_list = pd.DataFrame(positive_list)
        # using len(length) function for counting
        print("Positive Sentiment:", '%.2f' % len(positive_list), end='\n')
        print("Neutral Sentiment:", '%.2f' % len(neutral_list), end='\n')
        print("Negative Sentiment:", '%.2f' % len(negative_list), end='\n')

        return news_title_list , news_summary_list, len(positive_list), len(neutral_list), len(negative_list)
