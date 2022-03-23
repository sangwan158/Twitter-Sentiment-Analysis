import streamlit as st
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor
from textblob import TextBlob
import numpy as np
import pandas as pd
import re
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
CONSUMER_KEY = ""
CONSUMER_SECRET = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET =""
auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# twitter client
class TwitterClient():
    def __init__(self,twitter_user=None):
        self.twitter_client=API(auth)
        self.twitter_user=twitter_user
    
    def get_twitter_client_api(self):
        return self.twitter_client
    
    #get last n tweets
    def get_user_timeline(self,num_tweets):
        ls=list()
        for tweet in Cursor(self.twitter_client.user_timeline,id=self.twitter_user).items(num_tweets):
            ls.append(tweet)
        return ls
    
    def get_friend_list(self,num_friends):
        friend_list=list()
        for friend in Cursor(self.twitter_client.friends,id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list
    
    def get_home_timeline_tweets(self,num_tweets):
        home_list=list()
        for hometimeline in Cursor(self.twitter_client.home_timeline,id=self.twitter_user).items(num_tweets):
            home_list.append(hometimeline)
        return home_list

        


# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        pass

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)        
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(track=hash_tag_list)


# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
          

    def on_error(self, status):
        #staus 420 refers when data method reach limit occurs
        if status==420:
            return False
        print(status)

class TweetAnalyzer():
    def tweets_to_df(self,tweets):
        df=pd.DataFrame(data=[tweet.text for tweet in tweets],columns=['tweets'])
        df['id']=np.array([tweet.id for tweet in tweets])
        df['date']=np.array([tweet.created_at for tweet in tweets])
        df['likes']=np.array([tweet.favorite_count for tweet in tweets])
        df['source']=np.array([tweet.source for tweet in tweets])
        df['retweets']=np.array([tweet.retweet_count for tweet in tweets])
        df['length']=np.array([len(tweet.text) for tweet in tweets])
        return df
    
    def clean_tweet(self,tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self,tweet):
        analysis=TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity>0:
            return 1
        elif analysis.sentiment.polarity==0:
            return 0
        else:
            return -1

def name_sentiment(qq):
    if qq==1:
        return "Positive"
    
    elif qq==0:
        return "Neutral"

    else:
        return "Negative"

if __name__ == '__main__':
    st.title("Sentiment Analysis of Tweets")
    st.sidebar.title("Sentiment Analysis of Tweets")
    st.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦")
    st.sidebar.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦")

    userId= st.sidebar.text_area("Enter the exact twitter handle of the Personality for analysis (without @)")
    st.sidebar.subheader("")
    # Authenticate using config.py and connect to Twitter Streaming API.
    #hash_tag_list = ["arvind kejriwal"]
    #fetched_tweets_filename = "tweets.txt"
    #twitter_streamer = TwitterStreamer()
    #twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)
    twitterclient=TwitterClient('KapilSharmaK9')
    print(twitterclient.get_friend_list(10))
    
    num_tweets = st.sidebar.slider("No. of tweets to do analysis", 10, 1000)
    twitter_client=TwitterClient()
    tweet_analyzer=TweetAnalyzer()
    api=twitter_client.get_twitter_client_api()
    tweets=api.user_timeline(screen_name=userId,count=num_tweets)
    df=tweet_analyzer.tweets_to_df(tweets)
    if st.sidebar.checkbox("Show last 5 tweets", False, key='1'):
        recent_tweets=df['tweets'].head(5)		
        st.write(recent_tweets)

    #to check what we can extract from tweet
    #print(dir(tweets[0]))

    #check id of tweet
    #print(tweets[0].id)

    #retweet count
    #print(tweets[0].retweet_count)
    print(df)

    #get avg length over all tweets
    #print(np.mean(df['len']))
    
    #Time series for number of likes
    
    choice = st.sidebar.multiselect('Pick time series', ('likes','retweets','length'), key='timeSeries')
    
    
    if len(choice) > 0:
        st.subheader("Time Series Graph")
        for i in choice:
            time_likes=pd.Series(data=df[i].values,index=df['date'])
            time_likes.plot(figsize=(16,4),legend=True,label=i)
        st.pyplot()
        

    #Time series for number of retweets
    #time_likes=pd.Series(data=df['retweets'].values,index=df['date'])  #index is x-axis
    #time_likes.plot(figsize=(16,4),legend=True,label="likes")
    #plt.show()

    #to merge two plots
    #time_likes=pd.Series(data=df['likes'].values,index=df['date'])  #index is x-axis
    #time_likes.plot(figsize=(16,4),legend=True,label="likes")
    #time_likes=pd.Series(data=df['retweets'].values,index=df['date'])  #index is x-axis
    #time_likes.plot(figsize=(16,4),legend=True,label="retweets")
    #plt.show()


    #sentiment analyzes
    df['sentiment']=np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])
    df['Analysis'] = df['sentiment'].apply(name_sentiment)
    #print(df.head(10))

    #make wordcloud
    
    st.sidebar.header("Word Cloud")
    word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
    if not st.sidebar.checkbox("Close", True, key='word_cloud'):
        st.subheader('Word cloud for %s sentiment' % (word_sentiment))
        #df=make_df(userId)
        sent=0
        if word_sentiment==1:
            sent=1
        elif word_sentiment==0:
            sent=0
        else:
            sent=-1 
        df = df[df['sentiment']==sent]
        #tweets = ' '.join(df['Tweets'])
        processed_words = ' '.join([tweets for tweets in df['tweets']])
        wordcloud = WordCloud(stopwords=STOPWORDS,  background_color='black', width=5000, height=10000).generate(processed_words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        st.pyplot()

    st.sidebar.subheader('Visualisation')
    if not st.sidebar.checkbox("Hide Bar Plot For Sentiment Analysis", True, key='bar_plot'):
        st.header('Bar Plot For Sentiment Analysis')
        st.write(sns.countplot(x=df["Analysis"],data=df))
        st.pyplot(use_container_width=True)

    if not st.sidebar.checkbox("Hide Bar Plot source of tweet", True, key='bar_plot'):
        st.header('Source of Tweet')
        st.write(sns.countplot(x=df["source"],data=df))
        st.pyplot(use_container_width=True)
