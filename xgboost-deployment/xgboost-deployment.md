# XGBoost template

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/xgboost-deployment/xgboost-tutorial/xgboost-deployment.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/xgboost-deployment/xgboost-tutorial/xgboost-deployment.ipynb){ .md-button .md-button--secondary}

Note: This notebook runs on Python 3.11 and uses UbiOps CLient Library 4.5.1.

In this notebook we will show you the following:

How to create a deployment that uses the XGBoost library to make predictions on the prices of houses based on some criteria about the house.

This example uses the House Sales in King County, USA Dataset. [Link to the dataset](https://kaggle.com/harlfoxem/housesalesprediction)


If you run this entire notebook after filling in your access token, the XGBoost deployment will be deployed to your UbiOps environment. You can thus check your environment after running to explore. You can also check the individual steps in this notebook to see what we did exactly and how you can adapt it to your own use case.

We recommend to run the cells step by step, as some cells can take a few minutes to finish. You can run everything in one go as well and it will work, just allow a few minutes for building the individual deployments.


## 1. Establishing a connection with your UbiOps environment¶

Add your API token and your project name. We provide a deployment name and deployment version name. Afterwards we initialize the client library. This way we can deploy the XGBoost model to your environment.


```python
API_TOKEN = "<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>"
PROJECT_NAME = "<INSERT PROJECT NAME IN YOUR ACCOUNT>"
DEPLOYMENT_NAME = "xgboost-deployment"
DEPLOYMENT_VERSION = "v1"

# Import all necessary libraries
import shutil
import os
import ubiops

client = ubiops.ApiClient(
    ubiops.Configuration(
        api_key={"Authorization": API_TOKEN}, host="https://api.ubiops.com/v2.1"
    )
)
api = ubiops.CoreApi(client)

# This will create a new local folder to use for deployment files later
os.mkdir("xgboost-deployment")
```

## 2. Creating the model

This example will be based on [this Kaggle page](https://kaggle.com/mburakergenc/predictions-with-xgboost-and-linear-regression) about making predictions with XGBoost and Linear Regression.

In this document we focus on deploying the model to UbiOps, rather than on developing a model. Without elaborating much, we train a simple XGBoost model and save the resulting file to our deployment package directory.

After running this cell you should see a comparision between the `scikit-learn` model and the `xgboost` model regarding the accuracy score and the RMSE (Root Mean Square Error).

Let us first install the python packages we will need for our model:



```python
!pip install ubiops==4.5.1
!pip install pandas==2.2.2
!pip install numpy==2.1.1
!pip install scikit-learn==1.5.1
!pip install scipy==1.14.1
!pip install xgboost==2.1.1
!pip install joblib==1.4.2
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
data = pd.read_csv(
    "https://storage.googleapis.com/ubiops/data/Deploying%20with%20popular%20DS%20libraries/xgboost_tutorial/kc_house_data.csv"
)

# Train a simple linear regression model
regr = linear_model.LinearRegression()
new_data = data[
    [
        "sqft_living",
        "grade",
        "sqft_above",
        "sqft_living15",
        "bathrooms",
        "view",
        "sqft_basement",
        "lat",
        "waterfront",
        "yr_built",
        "bedrooms",
    ]
]

X = new_data.values
y = data.price.values

# Create train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Train the model
regr.fit(X_train, y_train)

# Check how the sklearn model scores on accuracy on our test set
sklearn_score = regr.score(X_test, y_test)
# Print the score of the sklearn model (Not great)
print(f"Score of the sklearn model: {sklearn_score}")

# Calculate the Root Mean Squared Error
print(
    "RMSE of the sklearn model: %.2f"
    % math.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2))
)

# Let's try XGboost algorithm to see if we can get better results
xgb = xgboost.XGBRegressor(
    n_estimators=100,
    learning_rate=0.08,
    gamma=0,
    subsample=0.75,
    colsample_bytree=1,
    max_depth=7,
)

traindf, testdf = train_test_split(X_train, test_size=0.2)
# Train the model
xgb.fit(X_train, y_train)

# Make predictions using the xgboost model
predictions = xgb.predict(X_test)


# Check how the xgboost model scores on accuracy on our test set
xgboost_score = explained_variance_score(predictions, y_test)

print(f"Score of the xgboost model {xgboost_score}")

# Calculate the Root Mean Squared Error
print(
    "RMSE of the xgboost model: %.2f" % math.sqrt(np.mean((predictions - y_test) ** 2))
)


# Save the model to our empty deployment package directory
joblib.dump(xgb, "xgboost-deployment/xgboost_model.joblib")
print("XGBoost model built and saved successfully!")
```

## 3. Creating the XGboost deployment
Now that we have our model saved it is time to create a deployment in UbiOps that will make use of it.

In the cell below you can view the `deployment.py` which will take data about the house we wish to predict the price of. As you can see in the initialization step we load the model we created earlier, then in the request method we make use of it to make a prediction. The input for to this model is:

* data: a csv file with the house data to predict its price.

The output of this model is: 

* data: a csv file with prediced housing pricing based on the data that is available. 


```python
%%writefile xgboost-deployment/deployment.py

"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import pandas as pd
import numpy as np
import os
from joblib import load

class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.
        """

        print("Initialising xgboost model")

        XGBOOST_MODEL = os.path.join(base_directory, "xgboost_model.joblib")
        self.model = load(XGBOOST_MODEL)

    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        """
        print('Loading data')
        input_data = pd.read_csv(data['data'])
        
        print("Prediction being made")
        prediction = self.model.predict(input_data.values)
        
        # Writing the prediction to a csv for further use
        print('Writing prediction to csv')
        pd.DataFrame(prediction).to_csv('prediction.csv', header = ['house_prices'], index_label= 'index')
        
        return {
            "prediction": 'prediction.csv'
        }

```


```python
%%writefile xgboost-deployment/requirements.txt

pandas==2.2.2
numpy==2.1.1
scikit-learn==1.5.1
scipy==1.14.1
xgboost==2.1.1
joblib==1.4.2
```

## 4. Deploying to UbiOps

Now we have all the pieces we need to create our deployment on UbiOps. In the cell below a deployment is being created, then a version of the deployment is created and the deployment code is zipped and uploaded to that version.



### Create the deployment


```python
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description="XGBoost deployment",
    input_type="structured",
    output_type="structured",
    input_fields=[
        {"name": "data", "data_type": "file"},
    ],
    output_fields=[{"name": "prediction", "data_type": "file"}],
    labels={"demo": "xgboost"},
)

api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
```

### Create a deployment version


```python
import ubiops.utils

version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment="python3-11",
    instance_type_group_name="512 MB + 0.125 vCPU",
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=1800,  # = 30 minutes
    request_retention_mode="full",  # we don't need request storage in this example
)

api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=version_template
)
```

### Package and upload the code


```python
# Zip the deployment package
shutil.make_archive("xgboost-deployment", "zip", ".", "xgboost-deployment")

# Upload the zipped deployment package
file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file="xgboost-deployment.zip",
)

# Wait for the deployment version to be ready
ubiops.utils.wait_for_deployment_version(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
)
```

## 5. Making a request to the deployment
You can navigate to the Web App and take a look in the user interface at what you have just built. If you want you can create a request to the XGBoost deployment using the code below, the sample data used is a small test subset(100 elements) from the original data. 

Since the data file will be processed in the UbiOps online environment, it needs to be uploaded to a bucket in our environment first. This ensures that the file can be accessed and used in the code.

### Prepare the data


```python
from ubiops import utils
import os

# Sample 100 elements from the original parsed dataset
sample_data = new_data.sample(n=100)

# Create a filepath to store the sample data at
current_dir = os.path.dirname(os.path.abspath("xgboost-deployment.ipynb"))
file_path = os.path.join(current_dir, "sample_data.csv")

# Reformat the sample data to a csv
sample_data.csv = sample_data.to_csv(file_path, index=False)

# Create the URI
file_uri = utils.upload_file(
    client=client,
    file_path=file_path,
    project_name=PROJECT_NAME,
    file_name="sample_data.csv",
    bucket_name="default",
)
```

### Make a request


```python
# Turn the sample data into dictionary format
input_data = {"data": file_uri}

# Use the previously established api connection to create a request
request = api.deployment_version_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=input_data,
)
```

#### All done! Let's close the client properly.


```python
client.close()
```

#### Exploring further

So there we have it! We have created a deployment and using the XGBoost  library. You can use this notebook to base your own deployments on. Just adapt the code in the deployment packages and alter the input and output fields as you wish and you should be good to go. 

For any questions, feel free to reach out to us via the customer service portal: https://ubiops.atlassian.net/servicedesk/customer/portals
