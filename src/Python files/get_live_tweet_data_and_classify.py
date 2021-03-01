# Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob
import re
import json
import datetime
import traceback
import sys

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("outputfile", nargs='?', default="tweets_bitcoin_test.json")
parser.add_argument("errorfile", nargs='?', default="tweets_bitcoin_error_test.txt")

args = parser.parse_args()

# Variables that contains the user credentials to access Twitter API
access_token = "2668453056-fBmEtF7Krb9cTTpCPZQqFlA5Lrvx1zjJ5sOd8Zr"
access_token_secret = "KNWQCRxRsRdZFr35ioXJqumVZWFatlwiVq2b0c1CJwi3D"
api_key = "kfgQylJMz8fY2dfXTh2hlwFJa"
api_secret = "y7HqjLCdEfNgzQT22rhnNhSs3de7p0m6FqeGQKRfebOpNAjBKw"


class Tweet:
    def __init__(self, text_val, sentiment_text_val, polarity, created_on):
        self.text_val = text_val
        self.sentiment_text_val = sentiment_text_val
        self.polarity = polarity
        self.created_on = created_on

    def __str__(self):
        return 'Sentiment: {} {} \nText: {}\ncreated_on: {}\n'.format(self.sentiment_text_val, self.polarity, self.text_val,
                                                                      self.created_on)


# Listener that prints received tweets to console and also stores the polarity.
class TwitterListener(StreamListener):
    def __init__(self):
        self.my_list = []

    def on_data(self, data):

        tweet = json.loads(data)

        try:
            if 'retweeted_status' in tweet:
                # Retweets Code
                if 'extended_tweet' in tweet['retweeted_status']:
                    # When extended beyond 140 Characters limit
                    tweet_text = tweet['retweeted_status']['extended_tweet']['full_text']
                else:
                    tweet_text = tweet['retweeted_status']['text']
            else:
                # Normal Tweets Code
                if 'extended_tweet' in tweet:
                    # Tweets over 140 Characters limit
                    tweet_text = tweet['extended_tweet']['full_text']
                else:
                    tweet_text = tweet['text']

            tweet_sentiment, polarity = self.get_tweet_sentiment(tweet_text)

            new_Tweet_Obj = Tweet(text_val=tweet_text,
                                sentiment_text_val=tweet_sentiment,
                                polarity=polarity,
                                created_on=datetime.datetime.strptime(tweet['created_at'],
                                                                      '%a %b %d %H:%M:%S %z %Y').strftime(
                                    '%Y-%m-%d-%H-%M-%S'))

            with open(args.outputfile, mode='a') as file:
                file.write('{},\n'.format(json.dumps(new_Tweet_Obj.__dict__)))

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            with open(args.errorfile, mode='a') as file:
                file.write('Error: {}\n\nTweet Info: {}\n--------------------------\n'.format(
                    repr(traceback.format_tb(exc_traceback)), tweet))

        return True

    def on_error(self, status):
        print(status)
        print('-----------------')

    def clean_tweet_data(self, tweet):
        '''
        Function to clean tweet by removing links and special characters
        using regex.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        '''
        Function to classify sentiments of tweets
        using Textblob's sentiment method
        '''
        # Creating TextBlob object of tweet text
        analysis = TextBlob(self.clean_tweet_data(tweet))

        # Setting sentiment
        sentiment = None
        if analysis.sentiment.polarity > 0:
            sentiment = 'positive'
        elif analysis.sentiment.polarity == 0:
            sentiment = 'neutral'
        else:
            sentiment = 'negative'

        return sentiment, analysis.sentiment.polarity

    def convert_sentiment_to_emoticon(self, sentiment_text_val):
        if sentiment_text_val == 'positive':
            return '✅'
        elif sentiment_text_val == 'negative':
            return '❌'
        else:
            return '🤷'


if __name__ == '__main__':
    # Code for Twitter authetification and the connection to Twitter Streaming API
    listener_obj = TwitterListener()
    auth = OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, listener_obj, tweet_mode='extended')

    # This line filter Twitter Streams
    stream.filter(track=['bitcoin', 'BTC'])