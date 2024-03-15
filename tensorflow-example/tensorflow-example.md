# Tensorflow template

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/tensorflow-example/tensorflow-ubiops-example){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/tensorflow-example/tensorflow-ubiops-example){ .md-button .md-button--secondary }

On this page we show you how to deploy a TensorFlow model to UbiOps. The TensorFlow model makes predictions on the fuel efficiency of late-1970s and early 1980s automobiles. This example uses [the classic Auto MPG dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data).

If you [download](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/tensorflow-example/tensorflow-ubiops-example){:target="_blank"} and run this entire notebook after filling in your access token, the TensorFlow deployment will be deployed to your UbiOps environment. You can check your environment after running to explore the results. You can also check the individual steps in this notebook to see what we did exactly and how you can adapt it to your own solution.

We recommend to run the cells step by step, as some cells can take a few minutes to finish. You can run everything in one go as well and it will work, just allow a few minutes for building the individual deployments.


## Establishing a connection with your UbiOps environment

We require an API Token with project editor rights to complete this tutorial. The final TensorFlow model ends up in a deployment with a default name (tensorflow-deployment) and a version. We define these parameters and connect to our API Client. Using this connection, we can interact with our project. Finally, we initiate a local empty directory that we can use to host our deployment files.


```python
API_TOKEN = '<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>' # Make sure this is in the format "Token token-code"
PROJECT_NAME = '<INSERT PROJECT NAME IN YOUR ACCOUNT>'
DEPLOYMENT_NAME = 'tensorflow-deployment'
DEPLOYMENT_VERSION = 'v1'

# Import all necessary libraries
import shutil
import os
import ubiops

client = ubiops.ApiClient(ubiops.Configuration(api_key={'Authorization': API_TOKEN}, 
                                               host='https://api.ubiops.com/v2.1'))
api = ubiops.CoreApi(client)

# Initiate a local directory
os.mkdir('tensorflow_deployment_package')
```

# Creating the model

This example is based on the [regression tutorial from tensorflow](https://tensorflow.org/tutorials/keras/regression#get_the_data).

On this page we focus on deploying the model to UbiOps rather than on developing a model. Without elaborating much, we train a simple model and save the resulting file to our deployment package directory.

Let us first install the Python packages we need for this model.


```python
import sys 
!pip install pandas
!pip install numpy
!pip install tensorflow
```


```python
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Predict the fuel efficiency of late-1970s and early 1980s automobiles

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Load data
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv('https://storage.googleapis.com/ubiops/data/Deploying%20with%20popular%20DS%20libraries/tensorflow-example/auto-mpg.csv', names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()

# Drop all but the horsepower and mpg columns
dataset = dataset[['Horsepower', 'MPG']]

# Drop unknown value rows
dataset = dataset.dropna()

# Split into train and test set 80-20
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Checking how our data structure looks like
print('Data:')
print(train_dataset.describe().transpose())

# Separate the target value, the "label", from the features. This label is the value that you will train the model to predict.
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')


# Create the horsepower Normalization layer:
horsepower = np.array(train_features)
horsepower_normalizer = preprocessing.Normalization(input_shape=[1,])
horsepower_normalizer.adapt(horsepower)

# Build the sequential model
model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])


# Configure training procedure
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Train the model using the prepared data
history = model.fit(
    train_features, train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2
)

# Calculate mean absolute error
mae = model.evaluate(
    test_features,
    test_labels,
    verbose=0

)

print(f'mean absolute error of the model is is {mae}')

# Save the model using h5 format for use within our deployment to make predictions
# Note that this will save the model to the folder where our model is 
model.save("tensorflow_deployment_package/tensorflow_model.h5")
print('Model created and saved successfully!')
```

## Creating the tensorflow deployment
Now that we have saved our model it is time to create a deployment in UbiOps that will make use of it.

In the cell below the deployment.py which will take the data we wish to predict the MPG for. As you can see in the initialization step we load the model we created earlier, then in the request method we make use of it to make a prediction. The input to this model is a csv file with the data to predict the MPG (mile per gallon)

```python
os.mkdir("tensorflow_deployment_package")
```


```python
%%writefile tensorflow_deployment_package/deployment.py
"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
import pandas as pd
from tensorflow.keras.models import load_model




class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.

        """

        print("Initialising the model")

        model_file = os.path.join(base_directory, "tensorflow_model.h5")
        self.model = load_model(model_file)


    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.

        """
        print('Loading data')
        input_data = pd.read_csv(data['data'])
        
        print("Prediction being made")
        prediction = self.model.predict(input_data)
        
        # Writing the prediction to a csv for further use
        print('Writing prediction to csv')
        pd.DataFrame(prediction).to_csv('prediction.csv', header = ['MPG'], index_label= 'index')
        
        return {
            "prediction": 'prediction.csv',
        }

```


```python
%%writefile tensorflow_deployment_package/requirements.txt

tensorflow==2.10.0
pandas==1.4.2
```


## Deploying to UbiOps

Now we have all the pieces we need to create our deployment on UbiOps. In the cell below we show how to create a deployment, how to create a version of the deployment and how to upload our deployment code to the deployment version.




```python
# Create the deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description='Tensorflow deployment',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'data', 'data_type':'file'}
    ],
    output_fields=[
        {'name':'prediction', 'data_type':'file'}
    ],
    labels={"demo": "tensorflow"}
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)

# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-9',
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
shutil.make_archive('tensorflow_deployment_package', 'zip', '.', 'tensorflow_deployment_package')

# Upload the zipped deployment package
file_upload_result =api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file='tensorflow_deployment_package.zip'
)
```

## All done! Let's close the client properly.


```python
client.close()
```

Note: This notebook runs on Python 3.9 and uses UbiOps CLient Library 3.15.0.

## Making a request and exploring further
You can go ahead to the Web App and take a look in the user interface at what you have just built. If you want you can create a request to the Tensorflow deployment using the ["dummy_data_to_predict.csv"](https://storage.googleapis.com/ubiops/data/Deploying%20with%20popular%20DS%20libraries/tensorflow-example/dummy_data_to_predict.csv). The dummy data is just the horsepower data. 

We have created a deployment that hosts a TensorFlow model. You can use this notebook  as a basis for your own TensorFlow deployments. Just adapt the code in the deployment packages and alter the input and output fields as you wish and you should be good to go.

For any questions, feel free to reach out to us via the customer service portal: https://ubiops.atlassian.net/servicedesk/customer/portals
