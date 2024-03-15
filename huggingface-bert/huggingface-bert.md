# Deploying a HuggingFace Transformer Model to UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/huggingface-bert/huggingface-bert/huggingface-bert.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/huggingface-bert/huggingface-bert/huggingface-bert.ipynb){ .md-button .md-button--secondary }

This notebook will help you create a cloud-based inference API endpoint for BERT, using UbiOps. The model we have is 
already pretrained and will be loaded from the Huggingface Transformers library. The workflow of this notebook can be used for other Huggingface models as well. We use the BERT model in this example, because it can run on a small CPU instance type.

In the following sections we will walk you through:

- Connecting with the UbiOps API client
- Creating a new UbiOps "deployment" with the BERT model
- How to call the BERT model with the model API


Let's get started!

## 1. Installing the UbiOps client library
To interface with UbiOps through code we need the UbiOps Python client library. In the following cell it will be installed.


```python
!pip install ubiops
```

## 2. Defining project info and setting up a connection

First, make sure you create an API token with `project-editor` permissions in your UbiOps project and paste it below. 
Also, fill in your corresponding UbiOps project name.

Once you have your project name and API token, paste them in the right spot in the following cell before running.


```python
import ubiops
from datetime import datetime
import os

API_TOKEN = '<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>' # Make sure this is in the format "Token token-code"
PROJECT_NAME = '<INSERT PROJECT NAME IN YOUR ACCOUNT>' # Fill in your project name here 

DEPLOYMENT_NAME = f"bert-base-uncased-{datetime.now().date()}"
DEPLOYMENT_VERSION = 'v1'
UBIOPS_STORAGE_BUCKET = 'default'

# Initialize client library
configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key['Authorization'] = API_TOKEN

# Establish a connection
client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
print(api.projects_get(PROJECT_NAME))
```

## 3. Preparing the deployment code

Now that we have defined our deployment in UbiOps, it is time to write our code to push it to UbiOps. Running the following cells will do that.


```python
!mkdir deployment_package
```

### a) Requirements.txt file

The `requirements.txt` file lists all the necessary packages that have to be installed in the environment. UbiOps will 
do this for you automatically.


```python
%%writefile deployment_package/requirements.txt
# This file contains package requirements for the deployment
# installed via PIP. Installed before deployment initialization

ubiops
numpy
torch==1.13.1
transformers
```

### b) Deployment.py file

For this example we create the code files and the deployment package directly from this notebook.

The `deployment.py` is the file that contains the code that will run on UbiOps each time a request is made. In this case the deployment is used to run the BERT model.


```python
%%writefile deployment_package/deployment.py

"""
The file containing the deployment code needs to be called 'deployment.py' and should contain a 'Deployment'
class a 'request' method.
"""

import os
import ubiops
from transformers import AutoTokenizer, BertForMaskedLM
import torch
import shutil


class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. Any code inside this method will execute when the deployment starts up.
        It can for example be used for loading modules that have to be stored in memory or setting up connections.
        """

        print("Initialising deployment")
        
        
        configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
        configuration.api_key['Authorization'] = os.environ['API_TOKEN']
        client = ubiops.ApiClient(configuration)
        api_client = ubiops.CoreApi(client)
        project_name = os.environ['PROJECT_NAME']

        tok_fn = "bert-base-uncased-tok"
        model_fn = "bert-base-uncased-model"
        
        try:
            ubiops.utils.download_file(
                        client,
                        project_name,
                        bucket_name="default", 
                        file_name=f"{tok_fn}.zip",
                        output_path=".",
                        stream=True,
                        chunk_size=8192
                        )

            shutil.unpack_archive(f"{tok_fn}.zip",f"./{tok_fn}", 'zip')
            print("Token file loaded from object storage")
            self.tokenizer = AutoTokenizer.from_pretrained(f"./{tok_fn}")

        except Exception as e:
            print(e)
            print("Tokenizer does not exist. Downloading from Hugging Face")

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            self.tokenizer.save_pretrained(f"./{tok_fn}")
            tok_dir = shutil.make_archive(tok_fn, 'zip', tok_fn)
            ubiops.utils.upload_file(client, project_name, f"{tok_fn}.zip", 'default')
        
        try:
            ubiops.utils.download_file(
                        client,
                        project_name,
                        bucket_name='default', 
                        file_name=f"{model_fn}.zip",
                        output_path='.',
                        stream=True,
                        chunk_size=8192
                        )

            shutil.unpack_archive(f"{model_fn}.zip",f"./{model_fn}", 'zip')
            print("Model file loaded from object storage")
            self.model = BertForMaskedLM.from_pretrained(f"./{model_fn}")

        except Exception as e:
            print(e)
            print("Model does not exist. Downloading from Hugging Face")

            self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
            self.model.save_pretrained(f"./{model_fn}")

            print("Storing model on UbiOps")
            model_dir = shutil.make_archive(model_fn, 'zip', model_fn)
            ubiops.utils.upload_file(client, project_name, f"{model_fn}.zip", "default")
            

    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        """

        print("Processing request")

        inputs = self.tokenizer(data["sentence"], return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # retrieve index of [MASK]
        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        result = self.tokenizer.decode(predicted_token_id)

        # here we set our output parameters in the form of a json
        return {"prediction": result}
```

## 4. Creating a UbiOps deployment

Now that we have our code ready, we can create a deployment.

We have set up this deployment in such a way that it expects a sentence as a string, with one word hidden with `[MASK]`. 
The output of the deployment will be the prediction for the value of the mask.

|Deployment input & output variables| | |
|--------------------|--------------|----|
| | **Variable name**| **Data type**|
| **Input fields** | sentence | string |
| **Output fields** | prediction | |


```python
# Create the deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    input_type='structured',
    output_type='structured',
    input_fields=[{'name': 'sentence', 'data_type': 'string'}],
    output_fields=[{'name': 'prediction', 'data_type': 'string'}]
)

api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
```

### Create a deployment version

Now we will create a version of the deployment. For the version we need to define the name, Python version, the type of instance (CPU or GPU) as well the size of the instance.

**For this we will use Python 3.10 with sufficient memory. Optionally you can run on a GPU which will speed up the inference, please [contact us](https://ubiops.com/contact-us/) if you want to enable this for your organization.**


```python
# Let's first create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-10',
    instance_type= '2048mb',
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=600, # = 10 minutes
    request_retention_mode='full'
)

api.deployment_versions_create(project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=version_template)

```

### Create environment variable

We need to create two environment variables, one for the API token and one for the project name. With these environment
variables we can upload the tokenizer and model from the initialization method in the `deployment.py`


```python
api_response = api.deployment_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=ubiops.EnvironmentVariableCreate(
        name='API_TOKEN',
        value=API_TOKEN,
        secret=True
))

api_response = api.deployment_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=ubiops.EnvironmentVariableCreate(
        name='PROJECT_NAME',
        value=PROJECT_NAME,
        secret=True
))
```

## 5. Package and upload the code

After defining the deployment and version, we can upload the code to UbiOps. We zip and upload the folder containing the
`requirements.txt` and `deployment.py` files. As we do this, UbiOps will build a container based on the settings above 
 and install all packages defined in our requirements file.

**Note** This step might take a few minutes, you can monitor the progress in the UbiOps WebApp by navigating to the 
deployment version and click the `logs` icon.


```python
# And now we zip our code (deployment package) and push it to the version

import shutil
zip_dir = shutil.make_archive("deployment_package", 'zip', 'deployment_package')

upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file='deployment_package.zip'
)
print(upload_response)
```

### Wait for the deployment to be ready

And now we just wait until the deployment is ready for use! It needs to build the container for a few minutes first.


```python
# Wait for the deployment version to be available


ubiops.utils.wait_for_deployment_version(api.api_client, PROJECT_NAME, DEPLOYMENT_NAME, DEPLOYMENT_VERSION, 
revision_id= upload_response.revision)

print("Deployment version is available")
```

## 6. Create a request to the model API on UbiOps to make predictions


```python
data = {
    "sentence": "Paris is the capital of [MASK].",
}

api.deployment_requests_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=data
).result
```

## 7. Wrapping up

And there you have it! We have succesfully created a deployment that uses a BERT model that was loaded from Huggingface.

Now all that is left to do is to close the connection to the UbiOps API.


```python
client.close()
```
