# MLFlow to UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/mlflow-conversion/mlflow-conversion){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/mlflow-conversion/mlflow-conversion){ .md-button .md-button--secondary }

This tutorial will guide you through the process of deploying an MLFlow model to UbiOps.  
For this tutorial, the [SKlearn Elasticnet Wine model](https://github.com/mlflow/mlflow/tree/6d0a9417dab9137e8a65af8a28d934b2dd41716c/examples/sklearn_elasticnet_wine) from the [MLFlow tutorial](https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html) will be used as an example.

### Method


In this tutorial we will we create an universal UbiOps deployment that loads any MLFlow model
that supports the Python function flavor.  
The Python function flavor is the default model interface for MLFlow Python models. This type of model can easily be loaded by calling the `mlflow.pyfunc.load_model` function. Moreover, scoring the model can also easily be done by calling the `predict` method. The following input data types are possible together with the `predict` method:

1. Pandas Dataframe
2. Numpy Array
3. Dict[`str`, numpy array]

UbiOps doesn't support these input types natively. Therefore, the input/output is converted to and from a JSON string. This means that the input/output of the UbiOps deployment
both need to be set to `String`.  
Furthermore, the requirements of the MLFlow model will be inserted into the `requirements.txt` file of the deployment.

## Tutorial structure

This tutorial will have the following structure:
- Install packages
- Train MLFlow model
- Convert model to UbiOps deployment
- Upload deployment to UbiOps
- Run Inference
  - Pandas Dataframe
  - Numpy
  - Dict

## Installing required packages

Some packages are needed to run the following notebook. They can be installed by running the code block below!


```python
!pip install "mlflow[extras]" -q
!pip install pyyaml -q
!pip install "ubiops >= 3.15" -q
```

## Train MLFlow model

Now, let's train our example MLFlow Model. This code snippet is directly copied from the [MLFlow Github examples](https://github.com/mlflow/mlflow/blob/959e8d90a13b62d755115501dede4531e157c1e7/examples/sklearn_elasticnet_wine/train.py),
with some adjustments to make it working in this notebook.

Note that the input/output signatures of the SKlearn model will not be saved in the MLFlow model!


```python
# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging

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
    l1_ratio = 1

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")
```

Try out different `alpha` and `l1_ratio` values to get different runs!

### Retrieve best run

Now it's time to retrieve the run with the smallest `RMSE`


```python
runs = mlflow.search_runs(order_by=["metrics.rmse ASC"])
best_run = runs.iloc[0]
```


```python
print(best_run)
```

## Convert to UbiOps deployment

We will now host this best MLFlow model inside a UbiOps deployment.

Let's first set some global variables!


```python
PATH_TO_MLFLOW_MODEL_ARTIFACT = os.path.join(best_run.artifact_uri, "model").replace("file://", "")

TOKEN = "<TOKEN>"
PROJECT_NAME = "<PROJECT NAME>"
DEPLOYMENT_NAME = "mlflow-auto-deployment"
VERSION_NAME = "v1"
```

### Creating files

Now it's time to create our deployment directory and add the right files to it so we can load our MLFlow model.  
UbiOps supports a `libraries` directory where dependencies can be included. This directory is added to the system `$PATH` variable, such that its contents can be easily imported.  
As earlier stated, UbiOps doesn't support the input types of the MLFlow Python flavor `predict` method natively. Therefore, we will add functions that will convert an input/output string to and from the input types in our libraries folder.  
All of the 3 different input types will be tested in the [Inference](#inference) section.


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
        print(f"Data type: {type(data_parsed)}")
        prediction = self.model.predict(data_parsed)
        return {"output": data_to_string(prediction)}

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


```python
%%writefile deployment_package/requirements.txt

mlflow
numpy
pandas
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
import os
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

if not TOKEN.startswith("Token "):
        TOKEN = "Token " + TOKEN

configuration = ubiops.Configuration()
# Configure API token authorization
configuration.api_key['Authorization'] = TOKEN
# Defining host is optional and default to "https://api.ubiops.com/v2.1"
configuration.host = "https://api.ubiops.com/v2.1"

api_client = ubiops.ApiClient(configuration)
```


```python
import ubiops
from ubiops import ApiException


def upload_mlflow_model_to_ubiops(api_client, project_name, deployment_name, version_name, deployment_zip,
                                  new_deployment=True, new_version=True):
    """
    Uploads a MLFlow model to UbiOps as a deployment

    :param ApiClient api_client: UbiOps API client
    :param str project_name: Project name
    :param str deployment_name: Name of existing/new deployment
    :param str version_name: Name of existing/new version
    :param str deployment_zip: Path to deployment zip file
    :param bool new_deployment: Whether to create a new deployment or use an existing deployment
    :param bool new_version: Whether to create a new version or use an existing version
    """

    core_api = ubiops.CoreApi(api_client)
    if new_deployment and not new_version:
        raise Exception("If new_deployment is True, new_version must be True as well")

    if new_deployment:
        # Check if deployment exists
        try:
            deployment = core_api.deployments_get(project_name=project_name, deployment_name=deployment_name)
            # Ask user if deployment should be overwritten
            overwrite = input(f"Deployment {deployment_name} already exists. Do you want to overwrite it? (Y/n) ")
            if overwrite.lower() == "y" or overwrite.lower() == "yes" or overwrite == "":
                print("Overwriting deployment")
                core_api.deployments_delete(project_name=project_name, deployment_name=deployment_name)
            else:
                raise Exception("Deployment already exists")

        except ApiException:
            pass

        deployment_template = ubiops.DeploymentCreate(
            name=deployment_name,
            description='MLFlow deployment',
            input_type='structured',
            output_type='structured',
            input_fields=[{'name': 'input', 'data_type': 'string'}],
            output_fields=[{'name': 'output', 'data_type': 'string'}],
            labels={"MLFlow": "auto-deployment"},
        )

        core_api.deployments_create(project_name=project_name, data=deployment_template)
    else:
        try:
            core_api.deployments_get(project_name=project_name, deployment_name=deployment_name)
        except ApiException:
            raise Exception("Deployment does not exist")

    if new_version:
        try:
            core_api.deployment_versions_get(
                project_name=project_name,
                deployment_name=deployment_name,
                version=version_name
            )
            # Ask user if version should be overwritten
            overwrite = input(f"Version {version_name} already exists. Do you want to overwrite it? (Y/n) ")
            if overwrite.lower() == "y" or overwrite.lower() == "yes" or overwrite == "":
                print("Overwriting version")
                core_api.deployment_versions_delete(
                    project_name=project_name,
                    deployment_name=deployment_name,
                    version=version_name
                )
            else:
                raise Exception("Version already exists")

        except ApiException:
            pass

        version_template = ubiops.DeploymentVersionCreate(
            version=version_name,
            environment='python3-10',
            instance_type='2048mb',
            maximum_instances=1,
            minimum_instances=0,
            maximum_idle_time=1800,  # = 30 minutes
            request_retention_mode='full'
        )

        core_api.deployment_versions_create(
            project_name=project_name,
            deployment_name=deployment_name,
            data=version_template
        )
    else:
        try:
            core_api.deployment_versions_get(
                project_name=project_name,
                deployment_name=deployment_name,
                version=version_name
            )
        except ApiException:
            raise Exception("Version does not exist")

    upload_response = core_api.revisions_file_upload(
        project_name=project_name,
        deployment_name=deployment_name,
        version=version_name,
        file=deployment_zip
    )
    print(upload_response)
```


```python
upload_mlflow_model_to_ubiops(
    api_client=api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version_name=VERSION_NAME,
    deployment_zip=deployment_zip
)
```

Let's wait for the deployment to be done!


```python
ubiops.utils.wait_for_deployment_version(
    client=api_client,
    project_name = PROJECT_NAME,
    deployment_name = DEPLOYMENT_NAME,
    version = VERSION_NAME
)
```

## Inference

Now it's time to run inference on the deployed MLFlow model inside UbiOps. All of the 3 different input types will be showcased, namely:  

1. Pandas Dataframe
2. Numpy Array
3. Dict[`str`, numpy array]

Do note that option 2, giving a `Numpy array` as input, does not work when the signatures are logged for the MLFlow model.
Use a `Pandas Dataframe` or `Dict[str, numpy array]` instead in this case!

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
core_api = ubiops.CoreApi(api_client)

result = core_api.deployment_version_requests_create(
  project_name=PROJECT_NAME,
  deployment_name=DEPLOYMENT_NAME,
  version=VERSION_NAME,
  data={"input": data_pandas_string}    
)
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

### Numpy array

Let's do the same now, but with a numpy array as input!


```python
data_numpy = data_pandas.to_numpy()
data_numpy_string = data_to_string(data_numpy)

result = core_api.deployment_version_requests_create(
  project_name=PROJECT_NAME,
  deployment_name=DEPLOYMENT_NAME,
  version=VERSION_NAME,
  data={"input": data_numpy_string}    
)

print(f"Original data type: {type(data_numpy)}")
print(f"Output of UbiOps request is: {type(result.result['output'])}")
result_converted = string_to_data(result.result["output"])
print(f"Output after conversion: {result_converted}")
print(f"Type after conversion: {type(result_converted)}")
```

### Dict[`str`, numpy array]



```python
data_dict = {k: np.array(v) for k, v in data_pandas.to_dict(orient="list").items()}
data_dict_string = data_to_string(data_numpy)

result = core_api.deployment_version_requests_create(
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
