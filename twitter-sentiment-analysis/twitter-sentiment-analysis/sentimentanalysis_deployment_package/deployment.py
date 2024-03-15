#!/usr/bin/env python
# coding: utf-8

import os
import time
from datetime import timedelta, datetime

from textblob import TextBlob
from tqdm import tqdm

import pygsheets
import tweepy


class Deployment:

    def __init__(self, base_directory, context):

        # Load in the twitter secrets and tokens from the environment variables
        self.consumer_key = os.environ['CONSUMER_KEY']
        self.consumer_secret = os.environ['CONSUMER_SECRET']
        self.access_token = os.environ['ACCESS_TOKEN']
        self.access_token_secret = os.environ['ACCESS_TOKEN_SECRET']

        # Set up the connection to twitter
        self.twitter_api = self.setup_twitter()

        # Setup the connection to Google, using the environment variable for the GOOGLE_CREDENTIALS
        # This method assumes you have an environment variable loaded with the content of the service account
        # credentials json
        self.google_sheet = pygsheets.authorize(service_account_env_var='GOOGLE_CREDENTIALS')

        # Set the spreadsheet_id from the environment variables
        self.spreadsheet_id = os.environ['SPREADSHEET_ID']

        # Set the day of today
        self.today = datetime.today()

    def setup_twitter(self):
        """
        Use the Tweepy package to connect to the twitter API and return the connection object
        """

        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, retry_count=2, retry_delay=240, timeout=120)

        try:
            api.verify_credentials()
            print("Authentication Twitter OK")
        except tweepy.error.TweepError as e:
            print(f"Error during authentication: {e}")
            raise e

        return api

    def request(self, data):
        """
        Make the request by first collecting the tweets and sentiments of a day and a certain hashtag and then
        inserting them in a Google sheet
        """

        hashtag = data.get('hashtag', 'MlOps')  # If no hashtag is given, use MlOps
        day = data.get('day', 'yesterday')  # If no day is given, use 'yesterday'

        # Parse the user inputted day and retrieve the end date of the query ('until')
        day, until = self.parse_date(day=day)

        # Retrieve tweets from 'day' to 'until'
        texts = self.retrieve_tweets(hashtag=hashtag, day=day, until=until)

        # Determine the sentiment over the recovered tweets
        results = self.get_sentiment(texts=texts, day=day)

        # Append the values to the specified Google Sheet
        sheet = self.google_sheet.open_by_key(key=self.spreadsheet_id)
        # Open first worksheet of spreadsheet
        wk1 = sheet[0]
        # Values will be appended after the last non-filled row of the table without overwriting
        wk1.append_table(values=results, overwrite=False)

        return None

    def parse_date(self, day):
        """
        Parse the user inputted date to be of yyyy-mm-dd and return the day and until date
        """

        date_format = "%Y-%m-%d"

        if day == "yesterday":
            # Convert the day and until date to the correct string format
            day = (self.today - timedelta(days=1)).strftime(date_format)
            until = self.today.strftime(date_format)

        else:
            # Check if the given date is in the correct format and not longer than 7 days ago
            try:
                day = datetime.strptime(day, date_format)
                if day < (self.today - timedelta(days=8)):
                    raise ValueError
            except ValueError:
                raise Exception(
                    f"Input for day is incorrect, it should be in the format of yyyy-mm-dd and should be no longer "
                    f"than 7 days ago")

            # Convert the day and until date to the correct string format
            until = (day + timedelta(days=1)).strftime(date_format)
            day = day.strftime(date_format)

        return day, until

    def retrieve_tweets(self, hashtag, day, until):
        """
        Return the tweet texts with the hashtag 'hashtag' that were created in one day
        """

        texts = []
        print(f"Retrieving tweets between {day} and {until}")

        retry = 0
        done = False

        # Query the Twitter api for all tweets on the specified hashtag and day and add them to a list
        while not done:
            try:
                for tweet in tqdm(tweepy.Cursor(
                    self.twitter_api.search, q=hashtag, count=20, until=until, lang="en", result_type="populair"
                ).items()):
                    if tweet.created_at.strftime("%Y-%m-%d") == day:
                        texts.append(tweet.text)
                    done = True

            except tweepy.error.TweepError as e:
                # Retry after 60 seconds if the connection gets lost
                print(f"Something went wrong while querying for tweets: {e}")
                time.sleep(60)
                retry += 1
                if retry < 4:
                    # Only make a maximum of 3 retry attempts
                    print(f"Retry attempt: {retry}")
                    continue
                raise e

        print(f"{len(texts)} tweets retrieved")
        return texts

    @staticmethod
    def get_sentiment(texts, day):
        """
        Perform sentiment analysis over all retrieved tweets and return the overall results
        """

        print("Calculating sentiment")

        neutral_list = []
        positive_list = []
        negative_list = []

        for tweet in tqdm(texts):
            t = TextBlob(tweet).sentiment.polarity

            if t > 0.1:
                positive_list.append(t)
            elif t < -0.1:
                negative_list.append(t)
            else:
                neutral_list.append(t)

        print(f"Sentiment calculated over {len(texts)} tweets from day {day}")

        # Convert the day to the exact format necessary for the Tableau dashboard
        day = datetime.strptime(day, "%Y-%m-%d").strftime("%d-%m-%Y")
        result = [day, len(positive_list), len(neutral_list), len(negative_list)]

        print(f"Result: {result}")
        return result
