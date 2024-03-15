# Twitter sentiment analysis

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/twitter-sentiment-analysis/twitter-sentiment-analysis){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/twitter-sentiment-analysis/twitter-sentiment-analysis/twitter-sentiment-analysis.ipynb){ .md-button .md-button--secondary }

Integrate UbiOps with google sheets, to visualise the outcome of sentiment analysis on tweets. This notebook is an
extension of the blog post written on this topic, which can be found on [our website](https://ubiops.com/running-machine-learning-behind-tableau).


On this page we will show you a notebook that explains how you can: 
- Connect with the Twitter API to collect tweets.
- Run a sentiment analysis model on the collected tweets.
- Connect with your own Google sheet file to push results to.

Out of scope for this  page, but relevant, is the visualisation in Tableau (that reads from google sheets as a database).

If you run this entire notebook, which you can download here, after filling in the Twitter access tokens and the Google sheets id, all required connections are set up and the model will start building. You can check via your UbiOps WebApp when the deployment has finished building, so that it's ready for requests. 
You can also check the individual steps on this page to see what we did exactly and how you can adapt it to your own use case, e.g. changing the hashtag. 

If you want to run the notebook, we recommend to run the cells step by step, as some cells can take a some times to finish. You can run everything in one go as well and it will work, just allow a few minutes for building the deployment.


## Acces tokens and credentials
Before getting started, follow the next steps to get the necessary Twitter tokens and Google credentials: 
1. Create a [service user](https://cloud.google.com/iam/docs/creating-managing-service-accounts#creating) in Google.
2. Create [credentials](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating_service_account_keys) for the service user (called “keys” here).
3. [Share](https://www.youtube.com/watch?v=sVURhxyc6jE) the Google sheet with the Google service user account just like you would with a normal user: You hereby give it permission to edit your sheet.
4. A Twitter [developer account](https://developer.twitter.com/en/apply-for-access) and access to the Twitter API.

## Jupyter notebook environment
Install and import the required packages to connect with UbiOps.
Install the UbiOps package (requirements.txt) in the environment where you run this notebook. Then run the following cell.


```python
import os
import json
import shutil
import ubiops
```

## Establishing a connection with your UbiOps environment
Add your API token below. Afterwards we initialize the client library. This way we can call different functions that are required for this process.


```python
# Ensure the format is "Token token-code"
# Make sure the service user has the project-editor role assigned

configuration = ubiops.Configuration()
configuration.api_key['Authorization'] = 'Token SERVICE USER TOKEN'  # You should copy this token when you create a service user in your UbiOps project

client = ubiops.ApiClient(configuration)
api = ubiops.api.CoreApi(client)

api.service_status()

```


```python
# Let's now define the variables all at once so we can refer to them later. 
# Define all variables as a string (inside '') except for the `google_credentials_dictionary` which should be defined as a dictionary.

CONSUMER_KEY = 'INSERT HERE Twitter consumer key'
CONSUMER_SECRET = 'INSERT HERE Twitter consumer secret'
ACCESS_TOKEN = 'INSERT HERE Twitter access token'
ACCESS_TOKEN_SECRET = 'INSERT HERE Twitter secret token'

SPREADSHEET_ID = 'INSERT HERE the Google spreadsheet ID'  # See the last part of the URL of your google sheet

google_credentials_dictionary = {} #  INSERT HERE the EXACT contents of the google credentials.json as a python dictionary
GOOGLE_CREDENTIALS = json.dumps(google_credentials_dictionary)

project_name = 'INSERT HERE project name'  # This must equal your project name in UbiOps
deployment_name = 'twitter-sa'  # Free to write anything you like here
deployment_version = 'v1'  # Free to write anything you like here

hashtag = 'mlops' # Change the hashtag you'd like to analyse tweets for here. In this notebook and the example blog it's set to 'mlops'.

```

## Deploying to UbiOps
Establishing the connections with Twitter and Google, collecting the tweets and running the sentiment analysis and more is done in the deployment.py file provided here. We now call the UbiOps API to create a deployment with the right deployment package (a .zip file). It's loaded below using a magic function, simply for you to have a look. In case you want a more thorough explanation, please see the article on this use case. 


```python
%%writefile sentimentanalysis_deployment_package/deployment.py
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

```


```python
# Set up the deployment template
deployment_template = ubiops.DeploymentCreate(
    name=deployment_name,
    description='Analyses tweets and pushes the results to Tableau for further analysis',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'hashtag', 'data_type':'string'},
        {'name':'day', 'data_type':'string'}
    ],
    output_fields=[]  # This deployment doesn't need any output fields, as the results are written to the sheet
)

# Create the deployment
api.deployments_create(
    project_name=project_name,
    data=deployment_template
)
```

## Creating environment variables
To connect with the Twitter API and give UbiOps access to your google sheet in safe way, we will create environment variables for your UbiOps deployment. This way, you don't need to hard code the credentials in the deployment package, while still being able to use them in the deployment. When running the cell below, you will see that 6 'environment variables' start to appear. You can see them being loaded in, in the deployment.py.


```python
# Create deployment environment variable
api_response = api.deployment_environment_variables_create(
    project_name=project_name,
    deployment_name=deployment_name,
    data=ubiops.EnvironmentVariableCreate(
        name='GOOGLE_CREDENTIALS',
        value=GOOGLE_CREDENTIALS,
        secret=True
))
print(api_response)

api_response = api.deployment_environment_variables_create(
    project_name=project_name,
    deployment_name=deployment_name,
    data=ubiops.EnvironmentVariableCreate(
        name='CONSUMER_KEY',
        value=CONSUMER_KEY,
        secret=True
))
print(api_response)

api_response = api.deployment_environment_variables_create(
    project_name=project_name,
    deployment_name=deployment_name,
    data=ubiops.EnvironmentVariableCreate(
        name='CONSUMER_SECRET',
        value=CONSUMER_SECRET,
        secret=True
))
print(api_response)

api_response = api.deployment_environment_variables_create(
    project_name=project_name,
    deployment_name=deployment_name,
    data=ubiops.EnvironmentVariableCreate(
        name='ACCESS_TOKEN_SECRET',
        value=ACCESS_TOKEN_SECRET,
        secret=True
))

print(api_response)

api_response = api.deployment_environment_variables_create(
    project_name=project_name,
    deployment_name=deployment_name,
    data=ubiops.EnvironmentVariableCreate(
        name='ACCESS_TOKEN',
        value=ACCESS_TOKEN,
        secret=True
))

print(api_response)

api_response = api.deployment_environment_variables_create(
    project_name=project_name,
    deployment_name=deployment_name,
    data=ubiops.EnvironmentVariableCreate(
        name='SPREADSHEET_ID',
        value=SPREADSHEET_ID,
        secret=True
))
print(api_response)
```

## Create the version
Now we have the deployment environment variables defined, we can create a version and let UbiOps build the environment with our deployment in it.


```python
# Create the version template
version_template = ubiops.DeploymentVersionCreate(
    version=deployment_version,
    environment='python3-8',
    instance_type='256mb',
    minimum_instances=0,
    maximum_instances=2,
    maximum_idle_time=1500,
    request_retention_mode='none'  #  We set this to none as we do not need to save the requests
)

# Create the version
api.deployment_versions_create(
    project_name=project_name,
    deployment_name=deployment_name,
    data=version_template
)

# Zip the deployment package if it doesn't exist already
if not os.path.exists('sentimentanalysis_deployment_package.zip'):
    shutil.make_archive('sentimentanalysis_deployment_package', 'zip', '.', 'sentimentanalysis_deployment_package')

# Upload the zipped deployment package
file_upload_result = api.revisions_file_upload(
    project_name=project_name,
    deployment_name=deployment_name,
    version=deployment_version,
    file='sentimentanalysis_deployment_package.zip'
)
print(file_upload_result)
```

## Create a request schedule
The following cell is optional. It creates a request schedule that will run the deployment with pre-defined request data at a pre-defined time. This can be useful to run the model every day and to analyse how the sentiment has changed over time.


```python
# Optional: add a request schedule to the deployment to automate the data collection.
# The following schedule will make a request to the deployment every day at 08:00 AM CET time (CEST+02:00).

request_schedule_template = ubiops.ScheduleCreate(
    name=deployment_name,
    object_type='deployment',
    object_name=deployment_name,
    schedule="0 6 * * *",  # May be adjusted to your liking. Standard it's set to daily at 8 AM CET time.
    request_data={
        'day': 'yesterday', 
        'hashtag': hashtag  # You can change this to another hashtag if you wish 
    }
)

# Create the schedule
api.request_schedules_create(
    project_name=project_name,
    data=request_schedule_template
)
```

## All done! Let's close the client properly.


```python
api_client.close()
```

Note: This notebook runs on Python 3.6 and uses the UbiOps CLient Library 3.15.0.

