# MLFlow to UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/mlflow-conversion/mlflow-conversion/MlFlow-Conversion-Tutorial.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/mlflow-conversion/mlflow-conversion/MlFlow-Conversion-Tutorial.ipynb){ .md-button .md-button--secondary}

In this tutorial, we'll create a generic UbiOps deployment that can load any MLFlow model with the Python function flavor.
This flavor is the default MLFlow model interface in Python, making it possible to load models with `mlflow.pyfunc.load_model` 
and perform predictions using the `predict` method.

Each MLFlow experiment run outputs a model artifact, which contains the model and a `requirements.txt` file. We'll provide a script to package these 
into a UbiOps deployment, which can be deployed to UbiOps directly, or be extended with preprocessing or postprocessing
scripts.

Our model’s `predict` method can accept:

1. A Pandas DataFrame
2. A dictionary (`Dict[str, numpy.ndarray]`)

Since UbiOps only supports JSON-serializable inputs and outputs (not DataFrames or tensors), we’ll include guidance on 
converting these data types to and from JSON strings. The UbiOps deployment will be configured with input/output fields 
of datatype `String`.



## Tutorial structure

This tutorial will have the following structure:
- Install packages
- Train MLFlow model
- Convert the model artifact to a UbiOps deployment
- Upload deployment to UbiOps
- Run Inference
  - Pandas Dataframe
  - Dict

## Installing required packages

We will need the following packages to run the tutorial:


```python
%pip install -U mlflow[extras]
%pip install -U pyyaml 
%pip install -U ubiops
```

## Train MLFlow model

Now, let's train our example MLFlow Model. This code snippet is directly copied from the [MLFlow Github examples](https://github.com/mlflow/mlflow/blob/959e8d90a13b62d755115501dede4531e157c1e7/examples/sklearn_elasticnet_wine/train.py),
with some adjustments to make it work in this notebook.


```python
# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = 0.5
    l1_ratio = 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)
        input_example = np.array(test_x)[:5]

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)
```

Try out different `alpha` and `l1_ratio` values to get different runs!

### Retrieve best run

Now it's time to retrieve the run with the smallest `RMSE`


```python
runs = mlflow.search_runs(order_by=["metrics.rmse ASC"])
best_run = runs.iloc[0]
print(best_run)
```

## Convert to UbiOps deployment

We will now deploy the best MLFlow model to UbiOps.

Let's first set some global variables that will allow us to connect to our UbiOps project, and set the name and version
name of the deployment


```python
API_TOKEN = "<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>"
PROJECT_NAME = "<INSERT PROJECT NAME IN YOUR ACCOUNT>"
DEPLOYMENT_NAME = "mlflow-auto-deployment"
VERSION_NAME = "v1"

PATH_TO_MLFLOW_MODEL_ARTIFACT = os.path.join(best_run.artifact_uri, "model").replace("file://", "")
```

### Creating the deployment package template

Now it's time to create our deployment directory and add the right files to it so we can load our MLFlow model.  
UbiOps supports a `libraries` directory where dependencies can be included. This directory is added to the system `$PATH` 
variable, such that its contents can be easily imported.  

As mentioned in the intro, UbiOps does not support the input types of the MLFlow Python flavor `predict` method natively. 
Therefore, we will add functions that will convert an input/output string to and from the input types in our `libraries`
directory.  

Both input types will be tested in the [Inference](#inference) section.


```python
!mkdir deployment_package
!mkdir deployment_package/libraries
```


```python
%%writefile deployment_package/deployment.py

import mlflow
import numpy
import pandas
from convert_data import data_to_string, string_to_data


class Deployment:
    def __init__(self):
        print(mlflow.__version__)
        self.model = mlflow.pyfunc.load_model("./model")

    def request(self, data):
        data_parsed = string_to_data(data["input"])
        print(f"Input data type: {type(data_parsed)}")
        prediction = self.model.predict(data_parsed)
        return {"output": data_to_string(prediction)}

```


```python
%%writefile deployment_package/requirements.txt

mlflow
numpy
pandas
```


```python
%%writefile deployment_package/libraries/convert_data.py

import json

import numpy as np
import pandas as pd


def data_to_string(data):
    if isinstance(data, pd.DataFrame):
        return json.dumps(data.to_dict())
    elif isinstance(data, np.ndarray):
        return json.dumps(data.tolist())
    elif isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
        return json.dumps({k: v.tolist() for k, v in data.items()})
    else:
        raise ValueError("Unsupported data type")


def string_to_data(data_str):
    data_json = json.loads(data_str)
    if isinstance(data_json, dict):
        if all(isinstance(v, list) for v in data_json.values()):
            return {k: np.array(v) for k, v in data_json.items()}
        else:
            return pd.DataFrame.from_dict(data_json)
    elif isinstance(data_json, list):
        return np.array(data_json)
    else:
        raise ValueError("Unsupported data type")
```

## Conversion functions

The following function will convert your MLModel artifact to a UbiOps deployment.  
The following steps are executed inside the function:

1. A check is performed to see if the `python_function` is supported in the MLFlow model
2. The `requirements.txt` of the MLFlow artifact is copied to the UbiOps deployment `requirements.txt`
3. Other model files are copied to the deployment directory
4. The deployment directory will be zipped
5. The deployment directory will be deleted depending on the corresponding function input


```python
import shutil

import yaml


def convert_to_deployment_package(path_to_model_artifact, new_deployment_package_name, remove_directory=True):
    """
    Converts a MLFlow model to a deployment package that can be uploaded to UbiOps
    :param path_to_model_artifact: Path to the MLFlow model artifact
    :param new_deployment_package_name: Name of the new deployment package
    :param remove_directory: Whether to remove the deployment directory after zipping
    """

    # Check if python_function exists under flavors in the MLmodel file
    with open(f"{path_to_model_artifact}/MLmodel", "r") as f:
        mlflow_yaml = yaml.safe_load(f)
        if "python_function" not in mlflow_yaml["flavors"]:
            raise Exception("No python_function flavor found in MLmodel file")

    # Append requirements.txt from MLflow model to requirements.txt in deployment package at the beginning
    # Double packages don't matter, pip will just ignore them in this case
    with open(f"{path_to_model_artifact}/requirements.txt", "r") as f:
        requirements = f.readlines()
        with open(f"{new_deployment_package_name}/requirements.txt", "r+") as f2:
            content = f2.read()
            f2.seek(0)
            f2.write("".join(requirements) + "\n" + content)

    # Copy the model to the deployment package
    shutil.copytree(path_to_model_artifact, f"{new_deployment_package_name}/model")

    # Zip the deployment package including the directory
    archive_location = shutil.make_archive(new_deployment_package_name, "zip", base_dir=new_deployment_package_name)

    print("Deployment package created successfully")

    if remove_directory:
        shutil.rmtree(new_deployment_package_name)

    return archive_location
```


```python
deployment_zip = convert_to_deployment_package(
    path_to_model_artifact=PATH_TO_MLFLOW_MODEL_ARTIFACT,
    new_deployment_package_name="deployment_package",
    remove_directory=False
)
```

## Upload to UbiOps

The following function will create a deployment in UbiOps and uploads the deployment package to it.  
Don't hesitate to read through the function to see what's happening!


```python
import ubiops

configuration = ubiops.Configuration()
# Configure API token authorization
configuration.api_key['Authorization'] = API_TOKEN
# Defining host is optional and defaults to "https://api.ubiops.com/v2.1"
configuration.host = "https://api.ubiops.com/v2.1"

client = ubiops.ApiClient(configuration)
api_client = ubiops.CoreApi(client)
```


```python
# Create deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description='MLFlow deployment',
    input_type='structured',
    output_type='structured',
    input_fields=[{'name': 'input', 'data_type': 'string'}],
    output_fields=[{'name': 'output', 'data_type': 'string'}],
    labels={"MLFlow": "auto-deployment"},
)
api_client.deployments_create(project_name=PROJECT_NAME, data=deployment_template)

# Create version
version_template = ubiops.DeploymentVersionCreate(
    version=VERSION_NAME,
    environment='python3-11',
    instance_type_group_name='2048 MB + 0.5 vCPU',
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=1800,  # = 30 minutes
    request_retention_mode='full'
)
api_client.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=version_template
)

```


```python
# Upload deployment code
upload_response = api_client.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=VERSION_NAME,
    file=deployment_zip
)
print(upload_response)
```

Let's wait for the deployment to be done!


```python
ubiops.utils.wait_for_deployment_version(
    client=client,
    project_name = PROJECT_NAME,
    deployment_name = DEPLOYMENT_NAME,
    version = VERSION_NAME
)
```

## Inference

Now it's time to run inference on the deployed MLFlow model inside UbiOps. Both input types will be shown:  

1. Pandas Dataframe
2. Dict[`str`, numpy array]


The following functions will be used to convert every data type to/from a string, so every data type will be interpretable by UbiOps.


```python
import json

import numpy as np
import pandas as pd


def data_to_string(data):
    if isinstance(data, pd.DataFrame):
        return json.dumps(data.to_dict())
    elif isinstance(data, np.ndarray):
        return json.dumps(data.tolist())
    elif isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
        return json.dumps({k: v.tolist() for k, v in data.items()})
    else:
        raise ValueError("Unsupported data type")


def string_to_data(data_str):
    data_json = json.loads(data_str)
    if isinstance(data_json, dict):
        if all(isinstance(v, list) for v in data_json.values()):
            return {k: np.array(v) for k, v in data_json.items()}
        else:
            return pd.DataFrame.from_dict(data_json)
    elif isinstance(data_json, list):
        return np.array(data_json)
    else:
        raise ValueError("Unsupported data type")
```

### Pandas Dataframe

In order to get a Pandas dataframe from the sample, we'll be grabbing the first 3 samples of the training set!


```python
data_pandas = train_x[:3]
print(data_pandas)
```

Let's transfer our Pandas dataframe to a string, so we can make a request to our deployment!


```python
data_pandas_string = data_to_string(data_pandas)
print(data_pandas_string)
```

Now, let's send this string to our deployment!


```python
result = api_client.deployment_version_requests_create(
  project_name=PROJECT_NAME,
  deployment_name=DEPLOYMENT_NAME,
  version=VERSION_NAME,
  data={"input": data_pandas_string}    
)
print(result)
```

As we can see, we get a perfect output back!  
We can even convert the string back to an usable data type!


```python
print(f"Original data type: {type(data_pandas)}")
print(f"Output of UbiOps request is: {type(result.result['output'])}")
result_converted = string_to_data(result.result["output"])
print(f"Output after conversion: {result_converted}")
print(f"Type after conversion: {type(result_converted)}")
```

### Dict[`str`, numpy array]



```python
data_dict = {k: np.array(v) for k, v in data_pandas.to_dict(orient="list").items()}
data_dict_string = data_to_string(data_dict)

result = api_client.deployment_version_requests_create(
  project_name=PROJECT_NAME,
  deployment_name=DEPLOYMENT_NAME,
  version=VERSION_NAME,
  data={"input": data_dict_string}    
)

print(f"Original data type: {type(data_dict)}")
print(f"Output of UbiOps request is: {type(result.result['output'])}")
result_converted = string_to_data(result.result["output"])
print(f"Output after conversion: {result_converted}")
print(f"Type after conversion: {type(result_converted)}")
```

So that's it! We have now created a generic deployment template that we can use to host MLFlow models of Python function 
flavor, which can take multiple input format. This set-up serves as an example. You can always customize and extend the 
set-up. Feel free to reach out to our [Support channel](https://support.ubiops.com) if you want to have a discussion with our team
