# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:11:08 2019

@author: nicol
"""
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json

 
import pandas as pd

ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""
CONSUMER_KEY = ""
CONSUMER_SECRET = ""


# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        return auth

# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()    

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app() 
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
            #print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
          
    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)
        
    def on_status(self, status):
        tweet_list = []
        tweet = status._json
        self.file.write(json.dumps(tweet) + "\n")
        tweet_list.append(status)
        self.num_tweets += 1
        if self.num_tweets < 3000:
            return True
        else:
            return False
        self.file.close()


#FUnction execution
if __name__ == '__main__':
 
    # Authenticate using config.py and connect to Twitter Streaming API.
    hash_tag_list = ["eggboy","NewZealand", "EggBoy"]
    fetched_tweets_filename = "C:\\Users\\nicol\\Documents\\a_exams\\4_digital\\twitter\\tweets9_eggboy.txt"

    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)
    
results = []
with open("C:\\Users\\nicol\\Documents\\a_exams\\4_digital\\twitter\\tweets9_eggboy.txt") as inputfile:
    for line in inputfile:
        if line != "\n":
            results.append(line) 

text_list = []
user_list = []
retweeted_names_list = []
tweet_type_list = []
listed_count_list = []
followers_count_list = []
mentions_list = []
count_mentions_list = []
user_mentioned_list_oflists = []
for i in range(0,len(results)):
    tweet_line = results[i]
    
    text = tweet_line.split(',"text":"')[1].split('","')[0]
    text_list.append(text)
    
    user_name = tweet_line.split(',"screen_name":"')[1].split('","')[0]
    user_list.append(user_name)
    
    try:
        user_retweeted = text.split('RT @')[1].split(': ')[0]
        tweet_type = "RT"
    except:
        user_retweeted = user_name
        tweet_type = "Tweet"
        pass
    tweet_type_list.append(tweet_type)
    retweeted_names_list.append(user_retweeted)
    
    listed_count = tweet_line.split('"listed_count":')[1].split(',"')[0]
    listed_count_list.append(listed_count)
    
    followers_count = tweet_line.split('"followers_count":')[1].split(',"')[0]
    followers_count_list.append(followers_count)
    
    count_mentions = 0
    #mentions in tweets and RT
    if tweet_type == "Tweet":
        user_mentioned_list = []
        for i in range(0,len(text)):
            if text[i] == "@":
                user_mentioned = ""
                for j in range(i+1,len(text)):
                    if text[j] == " ":
                        break
                    else:
                        letter = text[j]
                        user_mentioned = user_mentioned + str(letter)
                user_mentioned_list.append(user_mentioned)
                user_mentioned_list_oflists.append(user_mentioned_list)
    elif tweet_type == "RT":
        user_mentioned_list = []
        text_afterRT = text.split(':')[1]
        user_mentioned_list = []
        for i in range(0,len(text_afterRT)):
            if text_afterRT[i] == "@":
                user_mentioned = ""
                for j in range(i+1,len(text_afterRT)):
                    if text_afterRT[j] == " ":
                        break
                    else:
                        letter = text_afterRT[j]
                        user_mentioned = user_mentioned + str(letter)
                user_mentioned_list.append(user_mentioned)
                user_mentioned_list_oflists.append(user_mentioned_list)

df_extraction = pd.DataFrame()
df_extraction['text'] = text_list
df_extraction['user'] = user_list
df_extraction['user_retweeted'] = retweeted_names_list
df_extraction['tweet_type'] = tweet_type_list
df_extraction['listed_count'] = listed_count_list
df_extraction['followers_count'] = followers_count_list


#managing mentions
mentioned_user_list = []
for i in user_mentioned_list_oflists:
    list_mention = i
    for j in range(0,len(list_mention)):
        mentioned_user = list_mention[j]
        mentioned_user_list.append(mentioned_user)

mentioned_unique = set(mentioned_user_list)
counter_list = []
for i in range(0,len(mentioned_user_list)):
    counter = i
    counter_list.append(counter)

mentions_df = pd.DataFrame()
mentions_df['id'] = counter_list
mentions_df['user_mentioned'] = mentioned_user_list

find_duplicate_mentions = mentions_df.groupby(['user_mentioned'], as_index=False)['id'].count()

#managing retweets
retweet_only = df_extraction[df_extraction.tweet_type == 'RT']
number_retweets = retweet_only['user_retweeted']
for i in range(0,len(number_retweets)):
    counter = i
    counter_list.append(counter)
check_retweets = pd.DataFrame()    
check_retweets['id'] = counter_list
check_retweets['user_retweeted'] = number_retweets

find_duplicate_retweets = check_retweets.groupby(['user_retweeted'], as_index=False)['id'].count()

#join final
join_retweets = pd.merge(df_extraction, find_duplicate_retweets, how='left', left_on=['user'], right_on=['user_retweeted'])
join_retweets.drop('user_retweeted_y', axis=1, inplace=True)
join_retweets.columns = ['text', 'user', 'user_retweeted', 'tweet_type', 'listed_count', 'followers_count', 'retweets_received']

join_retweets_mentions = pd.merge(join_retweets, find_duplicate_mentions, how='left', left_on=['user'], right_on=['user_mentioned'])
join_retweets_mentions.drop('user_mentioned', axis=1, inplace=True)
join_retweets_mentions.columns = ['text', 'user', 'user_retweeted', 'tweet_type', 'listed_count', 'followers_count', 'retweets_received', 'mentions_received']
        
## Change path if needed
join_retweets_mentions.to_excel("C:\\Users\\nicol\\Documents\\a_exams\\4_digital\\twitter\\egg_boy_extraction_003.xlsx")













 

