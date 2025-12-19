# Deploying a Stable Diffusion model to UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/huggingface-stable-diffusion/huggingface-stable-diffusion/stable_diffusion_deployment.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/huggingface-stable-diffusion/huggingface-stable-diffusion/stable_diffusion_deployment.ipynb){ .md-button .md-button--secondary }

This notebook will help you create a cloud-based inference API endpoint for Stable Diffusion, using UbiOps. The Stable Diffusion version we'll be using is already pretrained and will be loaded from the Huggingface StableDiffusion library. The model has been developed by CompVis.

In this notebook we will walk you through:

- Connecting with the UbiOps API client
- Creating a code environment for our deployment
- Creating a deployment for the Stable Diffusion model
- Calling the Stable Diffusion deployment API endpoint

Stable Diffusion is a text-to-image model. Therefore we will make a deployment which takes a text prompt as an input, and returns an image:


|Deployment input & output variables| | |
|--------------------|--------------|----|
| | **Variable name**| **Data type**|
| **Input fields** | prompt | string |
| **Output fields** | image | file |

Note that we deploy to a GPU instance by default . If you do not have GPUs available in your account, you can modify the code so that it runs on a CPU instance instead, by changing the instance type from `16384mb_t4` to `16384mb`.
Let's get started!


## 1. Connecting with the UbiOps API client
To use the UbiOps API from our notebook, we need to install the UbiOps Python client library, and some other packages that we will use for visualisation of the result


```python
!pip install --upgrade ubiops
```

To set up a connection with the UbiOps platform API we need the name of your UbiOps project and an API token with `project-editor` permissions.

Once you have your project name and API token, paste them below in the following cell before running.


```python
import ubiops
from datetime import datetime

API_TOKEN = '<API TOKEN>' # Make sure this is in the format "Token token-code"
PROJECT_NAME = '<PROJECT_NAME>'    # Fill in your project name here

DEPLOYMENT_NAME = f"stable-diffusion-{datetime.now().date()}"
DEPLOYMENT_VERSION = 'gpu-t4'

# Initialize client library
configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key['Authorization'] = API_TOKEN

# Establish a connection
client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
print(api.projects_get(PROJECT_NAME))
```

## 2. Creating a code environment for our deployment
UbiOps will take your Python code and runs it as a microservice in the cloud. The platform will create a secure container image for your code to run in. To build this image, we will define an `environment` in UbiOps with all the right dependencies installed to run the Stable Diffusion model. Later we will attach an `instance_type_group` (hardware) as well. `Environments` can be reused for different models and make the deployment process easy and controllable.

So let's first define an `Environment` on top which our model can run!

### Setting up the environment

Our environment code contains instructions to install dependencies.


```python
environment_dir = 'environment_package'
ENVIRONMENT_NAME = 'stable-diffusion-environment-gpu'
```


```python
%mkdir {environment_dir}
```

We first write a requirements.txt file. This contains the Python packages that we will use in our deployment code


```python
%%writefile {environment_dir}/requirements.txt
# This file contains package requirements for the environment
# installed via PIP.
diffusers
transformers
scipy
torch==1.13.0+cu117
accelerate
```

Next we add a `ubiops.yaml` to set a remote pip index. This ensures that we install a CUDA-compatible version of PyTorch. CUDA allows models to be loaded and to run GPUs.


```python
%%writefile {environment_dir}/ubiops.yaml
environment_variables:
- PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu117
```

Now we create a UbiOps `environment`. We select Python 3.11 with CUDA pre-installed as the `base environment` if we want to run on GPUs. If we run on CPUs, then we use `python3-11`.

Our additional dependencies are installed on top of this base environment, to create our new `custom_environment` called `stable-diffusion-environment`.


```python
api_response = api.environments_create(
        project_name=PROJECT_NAME,
        data=ubiops.EnvironmentCreate(
        name=ENVIRONMENT_NAME,
        #display_name=ENVIRONMENT_NAME,
        base_environment='python3-11-cuda', #use python3-11 when running on CPU
        description='Environment to run Stable Diffusion from Huggingface',
        )
    )
```

Package and upload the environment instructions.


```python
import shutil
training_environment_archive = shutil.make_archive(environment_dir, 'zip', '.', environment_dir)
api.environment_revisions_file_upload(
        project_name=PROJECT_NAME,
        environment_name=ENVIRONMENT_NAME,
        file=training_environment_archive
    )
```

## 3. Creating a deployment for the Stable Diffusion model

Now that we have created our code environment in UbiOps, it is time to write the actual code to run the Stable Diffusion model and push it to UbiOps.

As you can see we're uploading a `deployment.py` file with a `Deployment` class and two methods:
- `__init__` will run when the deployment starts up and can be used to load models, data, artifacts and other requirements for inference.
- `request()` will run every time a call is made to the model REST API endpoint and includes all the logic for processing data.

Separating the logic between the two methods will ensure fast model response times.


```python
deployment_code_dir = 'deployment_code'
```


```python
!mkdir {deployment_code_dir}
```


```python
%%writefile {deployment_code_dir}/deployment.py

"""
The file containing the deployment code needs to be called 'deployment.py' and should contain a 'Deployment'
class a 'request' method.
"""

import os
import torch
from diffusers import StableDiffusionPipeline
import torch
import shutil
import numpy as np

class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. Any code inside this method will execute when the deployment starts up.
        It can for example be used for loading modules that have to be stored in memory or setting up connections.
        """
        model_id = "runwayml/stable-diffusion-v1-5"
        gpu_available = torch.cuda.is_available()
        device = torch.device("cuda") if gpu_available else torch.device("cpu")

        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if gpu_available else torch.float32)
        self.pipe = self.pipe.to(device)

        print("Initialising deployment")


    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        """
        image = self.pipe(data["prompt"]).images[0]
        print("Saving result")
        image.save("result.png")
        # here we set our output parameters in the form of a json
        return {"image": "result.png"}

```

### Create a UbiOps deployment

Create a deployment. Here we define the in- and outputs of a model. We can create different deployment versions



```python
# Create the deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    input_type='structured',
    output_type='structured',
    input_fields=[{'name': 'prompt', 'data_type': 'string'}],
    output_fields=[{'name': 'image', 'data_type': 'file'}]
)

api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
```

### Create a deployment version

Now we will create a version of the deployment. For the version we need to define the name, the environment, the type of instance (CPU or GPU) as well the size of the instance.


```python
# Let's first create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment=ENVIRONMENT_NAME,
    instance_type_group_name='16384 MB + 4 vCPU + NVIDIA Tesla T4', # You can use '16384 MB + 4 vCPU' if you run on CPU
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=600, # = 10 minutes
    request_retention_mode='full'
)

api.deployment_versions_create(project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=version_template)

```

Package and upload the code




```python
# And now we zip our code (deployment package) and push it to the version

import shutil
deployment_code_archive = shutil.make_archive(deployment_code_dir, 'zip', deployment_code_dir)

upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file= deployment_code_archive
)
print(upload_response)
```

We can only send requests to our deployment version, after our environment has finished building. 

NOTE: Building the environment might take a while as we need to download and install all the packages and dependencies. We only need to build our environment once: next time that we spin up an instance of our deployment, we won't need to install all dependencies anymore. Toggle off `stream_logs` to not stream logs of the build process.


```python
ubiops.utils.wait_for_deployment_version(api.api_client, 
                                        project_name=PROJECT_NAME, 
                                        deployment_name=DEPLOYMENT_NAME, 
                                        version=DEPLOYMENT_VERSION,
                                        stream_logs = True)
```

## 4. Calling the Stable Diffusion deployment API endpoint

Our deployment is now ready to be requested! We can send requests to it via the `deployment-requests-create` or the `batch-deployment-requests-create` API endpoint. It is going to take some time before the request finishes. When our deployment first loads, a GPU node will need to spin up, and we will need to download the Stable Diffusion model from HuggingFace. Subsequent results to the deployment will be handled faster. We will use a batch request to kick off our instance. This way, we can stream the on-start logs, and monitor the progress of the request using the `ubiops.utils` library.


```python
data = [{
    "prompt": "cyberpunk eiffel tower",
}]

response = api.batch_deployment_requests_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=data
)
response_id = response[0].id 
print(response[0])
```


```python
ubiops.utils.wait_for_deployment_request(api.api_client, 
                                         project_name = PROJECT_NAME, 
                                         deployment_name= DEPLOYMENT_NAME, 
                                         request_id = response_id,
                                         stream_logs = True
                                         )
```

Now retrieve the result of our image and visualise it


```python
file_uri = api.deployment_requests_get(PROJECT_NAME, DEPLOYMENT_NAME, response_id).result["image"]

ubiops.utils.download_file(client,
                        PROJECT_NAME,
                        file_uri = file_uri,
                        output_path='result.png')
```


```python
from IPython.display import Image

# Provide the path to your image
image_path = 'result.png'

# Display the image
Image(filename=image_path)
```

So that's it! You now have your own on-demand, scalable Stable Diffusion model running in the cloud, with a REST API that you can reach from anywhere!
