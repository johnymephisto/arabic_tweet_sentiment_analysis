import configparser

import pandas as pd
import tweepy


class TweetManager(object):

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('SentimentAnalysisArabic/config.ini')

        # config.read('config.ini')

        auth = tweepy.OAuthHandler(config['TWITTER_AUTH']['API_key'], config['TWITTER_AUTH']['API_secret_key'])
        auth.set_access_token(config['TWITTER_AUTH']['Access_token'], config['TWITTER_AUTH']['Access_token_secret'])
        self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)

    def get_tweets(self, query, result_type, count, lang='ar'):
        tweets = tweepy.Cursor(self.twitter_api.search, q=query, count=count, lang=lang, result_type=result_type)

        data = [[tweet.created_at, tweet.text] for tweet in tweets.items(count)]

        return pd.DataFrame(data, columns=['created_at', 'tweet'])

    def get_tweets_dummy(self, query, result_type, count, lang='ar'):
        return pd.read_csv('download.csv')


def main():
    """
    To test the classifier
    """
    df = TweetManager().get_tweets_dummy('corona', count=100, result_type='popular')
    df


if __name__ == '__main__':
    main()
