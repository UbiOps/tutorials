# Applying hyperparameter tuning on an XGBoost model

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/xgboost-training/xgboost-training/xgboost-training.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/xgboost-training/xgboost-training/xgboost-training.ipynb){ .md-button }

In this example, we will show you how you can initiate multiple training runs that train an XGBoost model with different hyperparameter combinations. We do this by creating an `environment` in which our training job can run. Then we will 
define a `train.py` script that we can apply to our environment. The training script is based on the XGBoost tutorial, 
where the [kc_house_data](https://www.kaggle.com/datasets/shivachandel/kc-house-data) dataset is used to train an XGBoost 
model that predicts house prices. We will initiate several runs using different sets of hyperparameters. After all the 
runs have been completed, we will explain how you can look at the results using the WebApp.


The output of the script is a trained XGBoost model (`xgboost_model.joblib`) and the the accuracy (`xgboost_score`) of 
the model. 

## Set project variables and initialize UbiOps API Client
First, make sure you create an **[API token](https://ubiops.com/docs/organizations/service-users/)** with `project editor` permissions in your UbiOps project and paste it below. Also, fill in your corresponding UbiOps project name. 


```python
%pip install --upgrade ubiops
```


```python
from datetime import datetime
import yaml
import os
import ubiops

API_TOKEN = '<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>' # Make sure this is in the format "Token token-code"
PROJECT_NAME = '<INSERT PROJECT NAME IN YOUR ACCOUNT>'
BUCKET_NAME = 'default'
```


```python
configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key['Authorization'] = API_TOKEN

api_client = ubiops.ApiClient(configuration)
core_instance = ubiops.CoreApi(api_client=api_client)
training_instance = ubiops.Training(api_client=api_client)
print(core_instance.service_status())
```

Set-up a training instance in case you have not done this yet in your project. This action will create a base training deployment, that is used to host training experiments.


```python
training_instance = ubiops.Training(api_client=api_client)
try:
    training_instance.initialize(project_name=PROJECT_NAME)
except ubiops.exceptions.ApiException as e:
    print(f"The training feature may already have been initialized in your project:\n{e}")
```

## Make the training environment


```python
training_environment_dir = 'training_environment'
ENVIRONMENT_NAME = 'xgboost-training-env'
```


```python
%mkdir {training_environment_dir}
```


```python
%%writefile {training_environment_dir}/requirements.txt
pandas==1.5.2
scikit-learn==1.0.2
scipy==1.10.0
xgboost==1.3.1
ubiops==3.9.0
fsspec==2022.1.0
joblib
pathlib
```

Now zip the environment like you would zip a deployment package, and create an environment


```python
import shutil 
training_environment_archive = shutil.make_archive(f'{training_environment_dir}', 'zip', '.', f'{training_environment_dir}')

# Create experiment. Your environment is set-up in this step. It may take some time to run.

try:
    api_response = core_instance.environments_create(
        project_name=PROJECT_NAME,
        data=ubiops.EnvironmentCreate(
        name=ENVIRONMENT_NAME,
        display_name= 'XGBoost training',
        base_environment='python3-11',
        description='XGboost training',
        )
    )
    
    core_instance.environment_revisions_file_upload(
        project_name=PROJECT_NAME,
        environment_name=ENVIRONMENT_NAME,
        file=training_environment_archive
    )
except ubiops.exceptions.ApiException as e:
    print(e)

```

## Configure an experiment

You can run experiments in your new environment. The experiments can help segment series of training runs, and run on one dedicated instance type. You can perform multiple runs in parallel in your experiment.

In this example, note that you are required to have a bucket inside your project. This bucket will be used to store your training jobs and model callbacks. In case you want to continue without [creating a bucket](https://github.com/UbiOps/client-library-python/blob/master/docs/Files.md#buckets_creates), you can use the `default` bucket. This bucket is always automatically generated for every project.


```python
BUCKET_NAME = 'default'
EXPERIMENT_NAME = 'xgboost-training-tutorial'
```


```python
try:
    experiment = training_instance.experiments_create(
        project_name=PROJECT_NAME,
        data=ubiops.ExperimentCreate(
            instance_type_group_name='4096 MB + 1 vCPU',
            description='Train test experiment',
            name=EXPERIMENT_NAME,
            environment=ENVIRONMENT_NAME,
            default_bucket= BUCKET_NAME
        )
    )
except ubiops.exceptions.ApiException as e:
    print(e)
```

## Then create the training runs
Now that we have our training experiment set-up, we can initiate training runs. For this example we will initiate parallel
training runs.


```python
RUN_NAME = 'training-run'
RUN_SCRIPT = f'{RUN_NAME}.py'
```


```python
%%writefile {RUN_SCRIPT}

import pandas as pd
import xgboost
import math
import os
import ubiops
import joblib
import pathlib
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


def train(training_data, parameters, context):

    # All code inside this function will run when a call to the deployment is made.
    # Read the data into a data frame
    
    data = pd.read_csv(training_data)

    print("Data loaded ")

    new_data = data[['sqft_living','grade', 'sqft_above', 'sqft_living15','bathrooms','view','sqft_basement','lat','waterfront','yr_built','bedrooms']].values
    X = new_data
    print('X loaded in')
    target_data = data[['price']]
    y = target_data.values
    print("splitting data")
   
    # Create train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)
    print('training model')
    
    # Set up the parameters
    n_est = parameters['n_estimators']
    le_ra = parameters['learning_rate']
    subsam = parameters['subsample']

    xgb = xgboost.XGBRegressor(n_estimators = n_est,learning_rate = le_ra, gamma = 0, subsample = subsam,
                                colsample_bytree = 1, max_depth = 7)

    print('parameters have been setup')

    # Train the model
    xgb.fit(X_train,y_train)
    print('model_trained')

    # Make predictions using the xgboost model
    predictions = xgb.predict(X_test)
    print('predictions made')

    # Check how the xgboost model scores on accuracy on our test set
    xgboost_score = explained_variance_score(predictions,y_test)

    print(f'Score of the xgboost model {xgboost_score}')

    # Save the model
    joblib.dump(xgb, 'xgboost_model.pkl') 
    print('XGBoost model built and saved successfully!')

    return {
        'artifact': 'xgboost_model.pkl',
        'metrics': {'xgboost_score': xgboost_score}
    }
```

## Training data

For this example we will download the training dataset locally, so we can show you how you can use a local dataset in a training run.


```python
import urllib.request

url = 'https://storage.googleapis.com/ubiops/data/Deploying%20with%20popular%20DS%20libraries/xgboost_tutorial/kc_house_data.csv'
training_data= 'kc_house_data.csv'

urllib.request.urlretrieve(url, training_data)

print(f"File downloaded successfully to '{training_data}'.")

```

## Defining the parameters

As shown in the train.py, this model uses six parameters. For simplicity we will only apply the hyperparameters to three
of those parameters. 

After the run is completed you can navigate to the `Training` tab, click the `Evaluation` button and select the three runs we completed and compare the results. The metrics `n_estimators`, `learning_rate`, and `subsample` from all three runs with different sets of hyperparameters, can then be compared with eachother. Here, we notice that the second set of parameters (`"n_estimators": 150, "learning_rate": 0.12, "subsample": 0.75`) achieves the highest score.
 
Alternatively, you can go to the experiment of which you want to compare runs (in this case the xgboost-training
-tutorial), select all the runs you want to compare by checking the boxes and then click on the `Compare runs` button.


```python
run_parameters = [
    {
    "n_estimators": 100,
    "learning_rate": 0.08,
    "subsample": 0.5
    },
     {
    "n_estimators": 150,
    "learning_rate": 0.12,
    "subsample": 0.75
    },
    {
    "n_estimators": 200,
    "learning_rate": 0.16,
    "subsample": 1
    }
]
```


```python
for i, run_parameters in enumerate(run_parameters):
    run_template = ubiops.ExperimentRunCreate(
            name=f"run{i}estimators{run_parameters['n_estimators']}_learning_rate{run_parameters['learning_rate']}",
            description='Trying out a first run run with ',
            training_code= RUN_SCRIPT,
            training_data= training_data, #path to data
            parameters= run_parameters
    )
    training_instance.experiment_runs_create(
        project_name=PROJECT_NAME,
        experiment_name=EXPERIMENT_NAME,
        data= run_template,
        timeout=14400
    )
```

## Wrapping up

And there you have it! We have succesfully trained an XGBoost model using the training functionality of UbiOps. You can download and adapt the code in this notebook to your own use case.

Now all that is left to do is to close the connection to the UbiOps API.


```python
api_client.close()
```

And there you have it! We have successfully created a deployment that trains an XGboost model. If you want you can check out the full [Jupyter Notebook](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/xgboost-training/xgboost-training/xgboost-training.ipynb), fill in your API token and project name, and run it yourself to upload it to your own UbiOps environment.
