# Arthur, UbiOps and XGBoost 

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/arthurai/arthurai){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/arthurai/arthurai/arthurai.ipynb){ .md-button .md-button--secondary }


Note: This notebook runs on Python 3.8 and uses UbiOps CLient Library 3.15.0.

In this notebook we will show you the following:

- How to create a deployment that uses a built xgboost model to make predictions on the prices of houses based on some criteria about the house and deploy that on [UbiOps](https://ubiops.com/)
- How to integrate with the [Arthur](https://arthur.ai/) platform to monitor your machine learning deployments.

This example uses the House Sales in King County, USA Dataset. [Link to the dataset](https://kaggle.com/harlfoxem/housesalesprediction)


If you run this entire notebook after filling in your access tokens, the xgboost deployment will be deployed to your UbiOps and Arthur environments. You can thus check your environment after running to explore. You can also check the individual steps in this notebook to see what we did exactly and how you can adapt it to your own use case.

We recommend to run the cells step by step, as some cells can take a few minutes to finish. You can run everything in one go as well and it will work, just allow a few minutes for building the individual deployments.


## Establishing a connection with your UbiOps environment¶

Add your API token and your project name. We provide a deployment name and deployment version name. Afterwards we initialize the client library. This way we can deploy the XGBoost model to your environment.


```python
API_TOKEN = "<YOUR UBIOPS API TOKEN>" # Make sure this is in the format "Token token-code"
PROJECT_NAME = "<YOUR PROJECT>"

DEPLOYMENT_NAME = 'xgboost-arthur-deployment'
DEPLOYMENT_VERSION = 'v1'

# Import all necessary libraries
import shutil
import os
import ubiops

client = ubiops.ApiClient(ubiops.Configuration(api_key={'Authorization': API_TOKEN}, 
                                               host='https://api.ubiops.com/v2.1'))
api = ubiops.CoreApi(client)
```

# Creating the model

This example will be based on [this kaggle](https://kaggle.com/mburakergenc/predictions-with-xgboost-and-linear-regression) about making predictions with XGboost and Linear Regression.

Since this document will be focused on the deploying side of the ML process. We will not cover the development of the model in-depth and make use of the pre-trained model below.

After running this cell you should see a comparision between the `sklearn` model and the `xgboost` model regarding the accuracy score and the RMSE (Root Mean Square Error)


Let us first install the python packages we will need for our model


```python
!pip install sklearn
!pip install xgboost
!pip install numpy
!pip install pandas
!pip install joblib
```


```python
import numpy as np
import pandas as pd
import xgboost
import math
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import joblib

# Read the data into a data frame
data = pd.read_csv('kc_house_data.csv').astype(dtype={'id': str})

# Train a simple linear regression model
regr = linear_model.LinearRegression()
input_columns = ['sqft_living','grade', 'sqft_above', 'sqft_living15','bathrooms','view','sqft_basement','lat','waterfront','yr_built','bedrooms']

# Create train test sets

train_data, test_data = train_test_split(data, test_size=0.2)

X_train, y_train = train_data[input_columns].to_numpy(), train_data['price'].to_numpy()
X_test, y_test = test_data[input_columns].to_numpy(), test_data['price'].to_numpy()

# X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)
# Train the model
regr.fit(X_train, y_train)

# Check how the sklearn model scores on accuracy on our test set
sklearn_score = regr.score(X_test,y_test)
# Print the score of the sklearn model (Not great)
print(f'Score of the sklearn model: {sklearn_score}')

# Calculate the Root Mean Squared Error
print("RMSE of the sklearn model: %.2f"
      % math.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2)))

# Let's try XGboost algorithm to see if we can get better results
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

# Train the model
xgb.fit(X_train,y_train)

# Make predictions using the xgboost model
predictions = xgb.predict(X_test)


# Check how the xgboost model scores on accuracy on our test set
xgboost_score = explained_variance_score(predictions,y_test)

print(f'Score of the xgboost model {xgboost_score}')

# Calculate the Root Mean Squared Error
print("RMSE of the xgboost model: %.2f"
      % math.sqrt(np.mean((predictions - y_test) ** 2)))


#save model
joblib.dump(xgb, 'xgboost-deployment/xgboost_model.joblib') 
print('XGBoost model built and saved successfully!')
```

## Registering the Model with Arthur

We'll create a connection to Arthur and then define the model, using the training data to infer the model input schema.


```python
from arthurai import ArthurAI
from arthurai.common.constants import Stage, InputType, OutputType, ValueType
```


```python
ARTHUR_URL = "https://app.arthur.ai"
ARTHUR_ACCESS_KEY = "<YOUR ARTHUR API KEY>" # Fill this in

connection = ArthurAI(url=ARTHUR_URL, access_key=ARTHUR_ACCESS_KEY)
```


```python
# Define the model schema
arthur_model = connection.model(partner_model_id="UbiOps House Prices",
                                input_type=InputType.Tabular,
                                output_type=OutputType.Regression,
                                is_batch=True)

arthur_model.from_dataframe(train_data[input_columns], Stage.ModelPipelineInput)
arthur_model.add_regression_output_attributes({"price": "price_gt"}, value_type=ValueType.Float)
arthur_model.review()
```

The dataframe above represents how the model will look to Arthur, and the format of the data it will expect. Notice how it detected some columns as categorical (such as Waterfront and View).

Now we can save the model to Arthur, and store the Arthur Model ID to be used by our deployment


```python
arthur_model_id = arthur_model.save()
with open("xgboost-deployment/arthur-model-id.txt", 'w') as f:
    f.write(arthur_model_id)
```

Finally, we'll upload the data we used to train the model as a reference set. Future data sent to the model will be compared to this reference set, to measure how much it has drifted from the types of inputs the model was built from.


```python
ref_df = train_data[['price'] + input_columns].rename(columns={'price': 'price_gt'})
ref_df['price'] = xgb.predict(ref_df[input_columns].to_numpy())
ref_df
```


```python
arthur_model.set_reference_data(data=ref_df)
```

## Creating the XGboost deployment
Now that we have our model saved it is time to create a deployment in UbiOps that will make use of it.

In the cell below you can view the deployment.py which will take data about the house we wish to predict the price of. As you can see in the initialization step we load the model we created earlier, then in the request method we make use of it to make a prediction. Input to this model is:

* data: a csv file with the house data to predict its price.



```python
%%writefile xgboost-deployment/deployment.py
"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import pandas as pd
import numpy as np
import os
import datetime
import pytz
from uuid import uuid4
from joblib import load
from arthurai.core.decorators import log_prediction
from arthurai import ArthurAI


class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.

        :param str base_directory: absolute path to the directory where the deployment.py file is located
        :param dict context: a dictionary containing details of the deployment that might be useful in your code.
            It contains the following keys:
                - deployment (str): name of the deployment
                - version (str): name of the version
                - input_type (str): deployment input type, either 'structured' or 'plain'
                - output_type (str): deployment output type, either 'structured' or 'plain'
                - environment (str): the environment in which the deployment is running
                - environment_variables (str): the custom environment variables configured for the deployment.
                    You can also access those as normal environment variables via os.environ
        """

        print("Initialising xgboost model")

        XGBOOST_MODEL = os.path.join(base_directory, "xgboost_model.joblib")
        self.model = load(XGBOOST_MODEL)

        with open("arthur-model-id.txt", 'r') as f:
            print("Initializing Arthur connection")
            self.arthur_model = ArthurAI().get_model(f.read())
            print("Successfully retrieved Arthur model")


    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.

        :param dict/str data: request input data. In case of deployments with structured data, a Python dictionary
            with as keys the input fields as defined upon deployment creation via the platform. In case of a deployment
            with plain input, it is a string.
        :return dict/str: request output. In case of deployments with structured output data, a Python dictionary
            with as keys the output fields as defined upon deployment creation via the platform. In case of a deployment
            with plain output, it is a string. In this example, a dictionary with the key: output.
        """
        print('Loading data')
        batch = pd.read_csv(data['data']).astype({'id': str})
        batch_id = str(uuid4()).split('-')[-1]

        print("Predictions being made")
        batch['price'] = self.model.predict(batch.drop(columns=['id']).to_numpy())

        print("Sending batch to Arthur")
        inference_data = [{'inference_timestamp': datetime.datetime.now(pytz.utc),
                           'partner_inference_id': row['id'],
                           'batch_id': batch_id,
                           'inference_data': {k: row[k] for k in row.keys() if k != 'id'}}
                           for row in batch.to_dict(orient='records')]
        self.arthur_model.send_inferences(inference_data)
        
        # Writing the prediction to a csv for further use
        print('Writing prediction to csv')
        batch['price'].to_csv('prediction.csv', header = ['house_prices'], index_label= 'index')
        
        return {
            "prediction": 'prediction.csv'
        }

```


## Deploying to UbiOps

Now we have all the pieces we need to create our deployment on UbiOps. In the cell below a deployment is being created, then a version of the deployment is created and the deployment code is zipped and uploaded to that version.




```python
# Create the deployment

deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description='XGBoost deployment',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'data', 'data_type':'file'},
    ],
    output_fields=[
        {'name':'prediction', 'data_type':'file'},
    ],
    labels={'demo': 'xgboost'}
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)

# Add Arthur environment variables
api.deployment_environment_variables_create(project_name=PROJECT_NAME,
                                            deployment_name=DEPLOYMENT_NAME,
                                            data=ubiops.EnvironmentVariableCreate(name='ARTHUR_ENDPOINT_URL',
                                                                                  value=ARTHUR_URL,
                                                                                  secret=False))
api.deployment_environment_variables_create(project_name=PROJECT_NAME,
                                            deployment_name=DEPLOYMENT_NAME,
                                            data=ubiops.EnvironmentVariableCreate(name='ARTHUR_API_KEY',
                                                                                  value=ARTHUR_ACCESS_KEY,
                                                                                  secret=True))

# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-8',
    instance_type='512mb',
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='none' # we don't need request storage in this example
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=version_template
)

# Zip the deployment package
shutil.make_archive('xgboost-deployment', 'zip', '.', 'xgboost-deployment')

# Upload the zipped deployment package
file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file='xgboost-deployment.zip'
)

# Check if the deployment is finished building. This can take a few minutes
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=file_upload_result.revision
)
```

## Making a request and exploring further

Finally we'll generate some sample data from our test set to use in the Web UI. After running the cell below take a look at the generated CSV files in the `sample_data` folder: we'll generate three batches of sample data and three files containing the true prices, all identified by the dataset's unique row IDs.


```python
from os import makedirs
makedirs('./sample_data', exist_ok=True)

NUM_BATCHES = 3
for batch_num in range(1, NUM_BATCHES+1):
    # choose a random set of indices for this batch
    batch_size = int(np.random.normal(100, 30))
    indices = np.random.choice(np.arange(len(test_data)), batch_size)
    
    # write out the input data with the ID to a CSV
    test_data.iloc[indices][input_columns + ['id']].to_csv(f'./sample_data/sample_batch_{batch_num}.csv', index=False)
    
    # write out the ground truth with the ID to a CSV, renaming the column 'price' to the ground truth 'price_gt'
    (test_data.iloc[indices][['id', 'price']].rename(columns={'price': 'price_gt'})
         .to_csv(f'./sample_data/ground_truth_batch_{batch_num}.csv', index=False))

```

You can go ahead to the UbiOps Web App and take a look in the user interface at what you have just built. Check out the `sample_data` directory and try uploading the `sample_batch_1.csv` file. You can then download the generated `predictions.csv` but they'll also be logged with Arthur.

## Sending actuals

Finally, we'll tell Arthur what the true price values were, so that we can compute accuracy metrics. We can send this ground truth at the same time as predictions, but we'll demonstrate sending it after the fact to simulate the real-world experience of receiving the true label sometime in the future.




```python
import datetime, pytz

def send_batch_ground_truth(filename):
    df = pd.read_csv(filename).astype({'id': str})
    ground_truth_data = []
    for row in df.itertuples():
        ground_truth_data.append({'partner_inference_id': row.id,
                                  'ground_truth_timestamp': datetime.datetime.now(pytz.utc),
                                  'ground_truth_data': {
                                      'price_gt': row.price_gt
                                  }})
    arthur_model.update_inference_ground_truths(ground_truth_data)
```


```python
send_batch_ground_truth('./sample_data/ground_truth_batch_1.csv')
```


```python
# send_batch_ground_truth('./sample_data/ground_truth_batch_2.csv')
# send_batch_ground_truth('./sample_data/ground_truth_batch_3.csv')
```

## All done! Let's close the client properly.


```python
api_client.close()
```

## Wrapping up
That's it! We've walked through building a model, creating it as a deployment with UbiOps, registering the model with Arthur, and sending data. Head over to the [Arthur UI](https://app.arthur.ai) to see the data, predictions, and analysis.

You can use this notebook to base your own deployments on. Just adapt the code in the deployment packages and alter the input and output fields as you wish and you should be good to go. 

For any questions, feel free to reach out to UbiOps via the [customer service portal](https://ubiops.atlassian.net/servicedesk/customer/portals) or Arthur via the chat on [the homepage](https://arthur.ai).
