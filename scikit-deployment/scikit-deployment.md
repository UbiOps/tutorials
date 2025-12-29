
# Scikit template

**Note**: This notebook runs on Python 3.11 and uses UbiOps CLient Library 3.15.0.

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/scikit-deployment/scikit-deployment){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/scikit-deployment/scikit-deployment){ .md-button .md-button--secondary }

On this page we will show you the following:

- How to make a training pipeline in UbiOps which preprocesses the data and trains and tests a model using scikit
- How to make a production pipeline in UbiOps which takes in new data, processes it and feeds it to a trained model for prediction/classification

For this example we will use a diabetes dataset from Kaggle to create a KNN classifier to predict if someone will have diabetes or not. [Link to original dataset](https://kaggle.com/uciml/pima-indians-diabetes-database).

If you [download](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/scikit-deployment/scikit-deployment){:target="_blank"} and run this entire notebook after filling in your access token, the two pipelines and all the necessary models will be deployed to your UbiOps environment. You can thus check your environment after running to explore. You can also check the individual steps in this notebook to see what we did exactly and how you can adapt it to your own use case.

We recommend to run the cells step by step, as some cells can take a few minutes to finish. You can run everything in one go as well and it will work, just allow a few minutes for building the individual deployments.

## Establishing a connection with your UbiOps environment
Add your API token. Then we will provide a project name, deployment name and deployment version name. Afterwards we connect to the UbiOps API, which allows us to create deployments and pipelines in our project. This way we can deploy the two pipelines to your environment.


```python
API_TOKEN = "<INSERT YOUR TOKEN HERE>" # Make sure this is in the format "Token token-code"
PROJECT_NAME = "<INSERT PROJECT NAME>"
DEPLOYMENT_NAME = 'data-preprocessor'
DEPLOYMENT_VERSION = 'v1'

# Import all necessary libraries
import shutil
import os
import ubiops
import requests

client = ubiops.ApiClient(ubiops.Configuration(api_key={'Authorization': API_TOKEN}, 
                                               host='https://api.ubiops.com/v2.1'))
api = ubiops.CoreApi(client)
```

## Making a training pipeline

Our training pipeline will consist of two steps: preprocessing the data, and training a model.
For each of these two steps we will create a separate deployment in UbiOps. This way the processing step can be reused later in the deployment pipeline (or in other pipelines) and each block will be scaled separately, increasing speed.

### Preprocessing the data
In the cell below the deployment.py of the preprocessing block is loaded. In the request function you can see that the deployment will clean up the data for further use and output that back in the form of two csv files. 
The deployment has the following input:
- data: a csv file with the training data or with real data
- training: a boolean indicating whether we using the data for training or not. In the case this boolean is set to true the target outcome is split of of the training data.

The use of the boolean input "training" allows us to reuse this block later in a production pipeline. 

## We initiate three empty directories that we fill with our deployment codes


```python
os.mkdir("preprocessing_package")
os.mkdir("predictor_package")
os.mkdir("training_package")
```


```python
%%writefile preprocessing_package/deployment.py
"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.

        """

        print("Initialising My Deployment")

    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        
        """

        print("Processing request for My Deployment")
        #Load the dataset
        print("Loading data")
        diabetes_data = pd.read_csv(data["data"])
        # The data contains some zero values which make no sense (like 0 skin thickness or 0 BMI). 
        # The following columns/variables have invalid zero values:
        # glucosem bloodPressure, SkinThicknes, Insulin and BMI
        # We will replace these zeros with NaN and after that we will replace them with a suitable value.
        
        print("Imputing missing values")
        # Replacing with NaN
        diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
        # Imputing NaN
        diabetes_data['Glucose'].fillna(diabetes_data['Glucose'].mean(), inplace = True)
        diabetes_data['BloodPressure'].fillna(diabetes_data['BloodPressure'].mean(), inplace = True)
        diabetes_data['SkinThickness'].fillna(diabetes_data['SkinThickness'].median(), inplace = True)
        diabetes_data['Insulin'].fillna(diabetes_data['Insulin'].median(), inplace = True)
        diabetes_data['BMI'].fillna(diabetes_data['BMI'].median(), inplace = True)
        
        # If this deployment is used for training, the target column
        # needs to be split from the data
        if data["training"] == True:
            X = diabetes_data.drop(["Outcome"], axis = 1) 
            y = diabetes_data.Outcome
        else:
            X = diabetes_data
            y = pd.DataFrame([1])
            
            
        print("Scaling data")
        # Since we are using a distance metric based algorithm we will use scikits standard scaler to scale all the features to [-1,1]
        sc_X = StandardScaler()
        X =  pd.DataFrame(sc_X.fit_transform(X,),
                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
               'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # UbiOps expects JSON serializable output or files, so we convert the dataframes to csv
        X.to_csv('X.csv', index = False)
        y.to_csv('y.csv', index = False, header = False)

        return {
            "cleaned_data": 'X.csv', "target_data": 'y.csv'
        }

```


```python
%%writefile preprocessing_package/requirements.txt

pandas==1.3.5
numpy==1.21.5
scikit-learn==1.0.2
scipy==1.7.3
```

Now we create a deployment and a deployment version for the package in the cell above. 


```python
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description='Clean up data',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'data', 'data_type':'file'},
        {'name':'training', 'data_type':'bool'}
    ],
    output_fields=[
        {'name':'cleaned_data', 'data_type':'file'},
        {'name':'target_data', 'data_type':'file'}
    ],
    labels={'demo': 'scikit-deployment'}
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)

# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-11',
    instance_type_group_name='512 MB + 0.125 vCPU',
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
shutil.make_archive('preprocessing_package', 'zip', '.', 'preprocessing_package')

# Upload the zipped deployment package
file_upload_result1 = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file='preprocessing_package.zip'
)
```

The first model will now have been deployed to your UbiOps environment. Go ahead and take a look in the UI in the tab deployments to see it for yourself. 


### Training and testing

Now that we have the preprocessing deployment in UbiOps, we need a deployment that can take the output of the preprocessing step and train a KNN model on it. The code for this is in the "training_package" directory and can be seen in the next cell. We are going to perform the same steps as above to deploy this code in UbiOps.


```python
%%writefile training_package/deployment.py
"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.
        """

        print("Initialising My Deployment")

    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        """

        print("Processing request for My Deployment")
        # Load the dataset
        print("Loading data")
        
        X = pd.read_csv(data["cleaned_data"])
        y = pd.read_csv(data["target_data"], header = None)
        print(X.shape)
        print(y.shape)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

        
        # Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=7) 
        
        # Fit the model on training data
        knn.fit(X_train,y_train)
        
        # Get accuracy on test set. Note: In case of classification algorithms score method represents accuracy.
        score = knn.score(X_test,y_test)
        print('KNN accuracy: ' + str(score))
        
        # let us get the predictions using the classifier we had fit above
        y_pred = knn.predict(X_test)
                
        # Output classification report
        print('Classification report:')
        print(classification_report(y_test,y_pred))
        
        # Persisting the model for use in UbiOps
        with open('knn.joblib', 'wb') as f:
           dump(knn, 'knn.joblib')
        
        
        return {
            "trained_model": 'knn.joblib', "model_score": score
        }

```


```python
%%writefile training_package/requirements.txt

pandas==1.3.5
numpy==1.21.5
scikit-learn==1.0.2
scipy==1.7.3
joblib==1.1.0
```

Time to deploy this step to UbiOps.


```python
deployment_template_t = ubiops.DeploymentCreate(
    name='model-training',
    description='Trains a KNN model',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'cleaned_data', 'data_type':'file'},
        {'name':'target_data', 'data_type':'file'}
    ],
    output_fields=[
        {'name':'trained_model', 'data_type':'file'},
        {'name':'model_score', 'data_type':'double'}
    ],
    labels={'demo': 'scikit-deployment'}
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template_t
)

# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-11',
    instance_type_group_name='512 MB + 0.125 vCPU',
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='none' # we don't need request storage in this example
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name='model-training',
    data=version_template
)

# Zip the deployment package
shutil.make_archive('training_package', 'zip', '.', 'training_package')

# Upload the zipped deployment package
file_upload_result2 = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name='model-training',
    version=DEPLOYMENT_VERSION,
    file='training_package.zip'
)
```

Check if both deployments, preprocessing and training, are available for further use. We can only use the models inside a pipeline after they have been built.


```python
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=file_upload_result1.revision
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name='model-training',
    version=DEPLOYMENT_VERSION,
    revision_id=file_upload_result2.revision
)
```

## Creating a training pipeline

So right now we have two deployments: one cleaning up the input data and one using that data for training a model. We want to tie these two blocks together to create a workflow. We can use pipelines for that. Let's create a pipeline that takes the same input as the preprocessing block.


```python
training_pipeline_name = "training-pipeline"

pipeline_template = ubiops.PipelineCreate(
    name=training_pipeline_name,
    description='A simple pipeline that cleans up data and trains a KNN model on it.',
    input_type='structured',
    input_fields=[
        {'name':'data', 'data_type':'file'},
        {'name':'training', 'data_type':'bool'}
    ],
    output_type='structured',
    output_fields=[
        {'name':'trained_model', 'data_type':'file'},
        {'name':'model_score', 'data_type':'double'}
    ],
    labels={'demo': 'scikit-deployment'}
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```

We have a pipeline, now we just need to create a version and add our deployments to it.

**IMPORTANT**: If you get an error like: "error":"Version is not available: The version is currently in the building stage"
Your model is not yet available and still building. 
Check in the UI if your model is ready and then rerun the block below.


```python
training_pipeline_version = "v1"

pipeline_template = ubiops.PipelineVersionCreate(
    version=training_pipeline_version,
    request_retention_mode='full',
    objects=[
        # preprocessor
        {
            'name': DEPLOYMENT_NAME,
            'reference_name': DEPLOYMENT_NAME,
            'version': DEPLOYMENT_VERSION
        },
        # model-training
        {
            'name': 'model-training',
            'reference_name': 'model-training',
            'version': DEPLOYMENT_VERSION
        }
    ],
    attachments=[
        # start --> preprocessor
        {
            'destination_name': DEPLOYMENT_NAME,
            'sources': [{
                'source_name': 'pipeline_start',
                'mapping': [
                    {"source_field_name": 'data','destination_field_name': 'data'},
                    {"source_field_name": 'training','destination_field_name': 'training'}
                ]
            }]
        },
        # preprocessor --> model-training
        {
            'destination_name': 'model-training',
            'sources': [{
                'source_name': DEPLOYMENT_NAME,
                'mapping': [
                    {"source_field_name": 'cleaned_data','destination_field_name': 'cleaned_data'},
                    {"source_field_name": 'target_data','destination_field_name': 'target_data'}
                ]
            }]
        },
        # model-training -> pipeline end
        {
            'destination_name': 'pipeline_end',
            'sources': [{
                'source_name': 'model-training',
                'mapping': [
                    {"source_field_name": 'trained_model','destination_field_name': 'trained_model'},
                    {"source_field_name": 'model_score','destination_field_name': 'model_score'}
                ]
            }]
        }
    ]
)

api.pipeline_versions_create(project_name=PROJECT_NAME, pipeline_name=training_pipeline_name, data=pipeline_template)
```

## Training pipeline done!
If you check in your UbiOps account under pipeline you will find a training-pipeline with our components in it and connected. Let's make a request to it. You can also make a request in the UI with the "create direct request button".

This might take a while since the models will need a cold start as they have never been used before.


```python
training_pipeline_name = "training-pipeline"

csv = requests.get('https://storage.googleapis.com/ubiops/data/Deploying%20with%20popular%20DS%20libraries/sci-kit-deployment/diabetes.csv')

with open("diabetes.csv", "wb") as f:
    f.write(csv.content)

file_uri = ubiops.utils.upload_file(client, PROJECT_NAME, 'diabetes.csv')

data = {'data': file_uri, 'training': True}
pipeline_result = api.pipeline_version_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=training_pipeline_name,
    version=training_pipeline_version,
    data=data
)

print(pipeline_result)
```

## Predicting with the trained model

Our model is trained and ready. Now we still need to deploy a predictor to UbiOps that uses this model for predicting. 

I already have the code and the requirements ready that need to be deployed to UbiOps. However, the joblib file is still missing in this folder. We dont want to manually download the joblib file output from the training pipeline, but automatically put it in the deployment package for the predictor. After that we can zip up the folder and push it to UbiOps like we did with the previous two packages.


```python
%%writefile predictor_package/deployment.py
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

        print("Initialising KNN model")

        KNN_MODEL = os.path.join(base_directory, "knn.joblib")
        self.model = load(KNN_MODEL)

    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        """
        print('Loading data')
        input_data = pd.read_csv(data['data'])
        
        print("Prediction being made")
        prediction = self.model.predict(input_data)
        diabetes_instances = int(sum(prediction))
        
        # Writing the prediction to a csv for further use
        print('Writing prediction to csv')
        pd.DataFrame(prediction).to_csv('prediction.csv', header = ['diabetes_prediction'], index_label= 'index')
        
        return {
            "prediction": 'prediction.csv', "predicted_diabetes_instances": diabetes_instances
        }

```


```python
%%writefile predictor_package/requirements.txt

pandas==1.3.5
numpy==1.21.5
scikit-learn==1.0.2
scipy==1.7.3
joblib==1.1.0
```


```python
# We need to download the trained model joblib and put it in the predictor package directory
base_directory = os.path.dirname(os.path.abspath("scikit-deployment"))
output_path = os.path.join(base_directory, "predictor_package")

ubiops.utils.download_file(client, PROJECT_NAME, file_name='knn.joblib', output_path=output_path)
```


```python
# Now we need to zip the deployment package
shutil.make_archive('predictor_package', 'zip', '.', 'predictor_package')
```

## Deploying the KNN model
The folder is ready, now we need to make a deployment in UbiOps. Just like before.


```python
deployment_template = ubiops.DeploymentCreate(
    name='knn-model',
    description='KNN model for diabetes prediction',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'data', 'data_type':'file'},
    ],
    output_fields=[
        {'name':'prediction', 'data_type':'file'},
        {'name':'predicted_diabetes_instances', 'data_type':'int'}
    ],
    labels={'demo': 'scikit-deployment'}
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)

# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-11',
    instance_type_group_name='512 MB + 0.125 vCPU',
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='none' # we don't need request storage in this example
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name='knn-model',
    data=version_template
)

# Upload the zipped deployment package
file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name='knn-model',
    version=DEPLOYMENT_VERSION,
    file='predictor_package.zip'
)
```

Check if the deployment is ready for use


```python
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name='knn-model',
    version=DEPLOYMENT_VERSION,
    revision_id=file_upload_result.revision
)
```

## Creating the production pipeline



```python
prod_pipeline_name = "production-pipeline"

pipeline_template = ubiops.PipelineCreate(
    name=prod_pipeline_name,
    description="A simple pipeline that cleans up data and let's a KNN model predict on it.",
    input_type='structured',
    input_fields=[
        {'name':'data', 'data_type':'file'},
        {'name':'training', 'data_type':'bool'}
    ],
    output_type='structured',
    output_fields=[
        {'name':'prediction', 'data_type':'file'},
        {'name':'predicted_diabetes_instances', 'data_type':'int'}
    ],
    labels={'demo': 'scikit-deployment'}
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```

We have a pipeline, now we just need to create a version and add our deployments to it.

**IMPORTANT**: If you get an error like: "error":"Version is not available: The version is currently in the building stage"
Your model is not yet available and still building. 
Check in the UI if your model is ready and then rerun the block below.


```python
prod_pipeline_version = DEPLOYMENT_VERSION

pipeline_template = ubiops.PipelineVersionCreate(
    version=prod_pipeline_name,
    request_retention_mode='none',
    objects=[
        # Preprocessor
        {
            'name': DEPLOYMENT_NAME,
            'reference_name': DEPLOYMENT_NAME,
            'version': DEPLOYMENT_VERSION
        },
        # KNN model
        {
            'name': 'knn-model',
            'reference_name': 'knn-model',
            'version': DEPLOYMENT_VERSION
        }
    ],
    attachments=[
        # start --> preprocessor
        {
            'destination_name': DEPLOYMENT_NAME,
            'sources': [{
                'source_name': 'pipeline_start',
                'mapping': [
                    {"source_field_name": 'data','destination_field_name': 'data'},
                    {"source_field_name": 'training','destination_field_name': 'training'}
                ]
            }]
        },
        # preprocessor --> KNN model
        {
            'destination_name': 'knn-model',
            'sources': [{
                'source_name': DEPLOYMENT_NAME,
                'mapping': [
                    {"source_field_name": 'cleaned_data','destination_field_name': 'data'},
                ]
            }]
        },
        # KNN model --> pipeline end
        {
            'destination_name': 'pipeline_end',
            'sources': [{
                'source_name': 'knn-model',
                'mapping': [
                    {"source_field_name": 'prediction','destination_field_name': 'prediction'},
                    {"source_field_name": 'predicted_diabetes_instances','destination_field_name': 'predicted_diabetes_instances'}
                ]
            }]
        }
    ]
)

api.pipeline_versions_create(project_name=PROJECT_NAME, pipeline_name=prod_pipeline_name, data=pipeline_template)
```

## All done! Let's close the client properly.


```python
client.close()
```

## Making a request and exploring further
You can go ahead to the Web App and take a look in the user interface at what you have just built. If you want you can create a request to the production pipeline using the ["dummy_data_for_predicting.csv"](https://storage.googleapis.com/ubiops/data/Deploying%20with%20popular%20DS%20libraries/sci-kit-deployment/dummy_data_for_predicting.csv) and setting the "training" input to "False". The dummy data is just the diabetes data with the Outcome column chopped of. 

So there we have it! We have made a training pipeline and a production pipeline using the scikit learn library. You can use this notebook to base your own pipelines on. Just adapt the code in the deployment packages and alter the input and output fields as you wish and you should be good to go. 

For any questions, feel free to reach out to us via the [customer service portal](https://ubiops.atlassian.net/servicedesk/customer/portals)
