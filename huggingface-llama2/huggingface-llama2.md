# Deploying Llama 2 to UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/huggingface-llama2/huggingface-llama2/llama_2_deployment.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/huggingface-llama2/huggingface-llama2/llama_2_deployment.ipynb){ .md-button .md-button--secondary }

This tutorial will help you create a cloud-based inference API endpoint for the [Llama-2-7B-HF model](https://huggingface.co/meta-Llama/Llama-2-7b-chat-hf),
using UbiOps. The Llama 2 version we will be using is already pretrained and will be 
loaded from the Huggingface [Meta-Llama](https://huggingface.co/meta-Llama) library. The model has been developed by Meta AI.

Llama 2 is a collection of models, ranging from 7 billion to 70 billion parameters. The version that is used in this tutorial
is the fine-tuned version of Llama 7B. it has been optimized for use cases involving dialogues.

Note that you will need to have an HF token if you want to download a Llama 2 model from Huggingface. You can apply for one
[here](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform).

In this tutorial we will walk you through:

1. Connecting with the UbiOps API client
2. Creating a code environment for our deployment
3. Creating a deployment for the Llama 2 model
4. Calling the Llama 2 deployment API endpoint

Llama 2 is a text-to-text model. Therefore we will make a deployment which takes a text prompt as an input, and returns
a response:

|Deployment input & output variables| **Variable name** |**Data type** |
|--------------------|--------------|----
| **Input fields** | prompt | string |
| **Output fields** | response | string |

Note that we deploy to a GPU instance by default, which are not accessible in every project. You can 
[contact us](https://ubiops.com/contact-us/) about this.
Let's get started!


## 1. Connecting with the UbiOps API client
To use the UbiOps API from our notebook, we need to install the UbiOps Python client library.

```python
!pip install --upgrade ubiops
```

To set up a connection with the UbiOps platform API we need the name of your UbiOps project and an API token with `project-editor` permissions.

Once you have your project name and API token, paste them below in the following cell before running.


```python
import ubiops
from datetime import datetime

API_TOKEN = "<API TOKEN>"  # Make sure this is in the format "Token token-code"
PROJECT_NAME = "<PROJECT_NAME>"  # Fill in your project name here

DEPLOYMENT_NAME = f"llama-2-7b-{datetime.now().date()}"
DEPLOYMENT_VERSION = "v1"

# Initialize client library
configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key["Authorization"] = API_TOKEN

# Establish a connection
client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
print(api.projects_get(PROJECT_NAME))
```

## 2. Setting up the environment

Our environment code contains instructions to install dependencies.


```python
environment_dir = "environment_package"
ENVIRONMENT_NAME = "llama-2-environment"
```


```python
%mkdir {environment_dir}
```

We first write a `requirements.txt` file, which contains the Python packages that we will use in our deployment code


```python
%%writefile {environment_dir}/requirements.txt
# This file contains package requirements for the environment
# installed via PIP.
diffusers
transformers
scipy
torch==1.13.1
accelerate
huggingface-hub
ubiops
```

Next we add a `ubiops.yaml` to set a remote pip index. This ensures that we install a CUDA-compatible version of PyTorch. 
CUDA allows models to be loaded and to run GPUs.


```python
%%writefile {environment_dir}/ubiops.yaml
environment_variables:
- PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu110
```

Now we create a UbiOps `environment`. We select Python3.9 with CUDA pre-installed as the `base environment` if we want 
to run on GPUs. If we run on CPUs, then we use `python3-9`.

Our additional dependencies are installed on top of this base environment, to create our new `custom_environment` 
called `llama-2-environment`.


```python
api_response = api.environments_create(
    project_name=PROJECT_NAME,
    data=ubiops.EnvironmentCreate(
        name=ENVIRONMENT_NAME,
        # display_name=ENVIRONMENT_NAME,
        base_environment="python3-9-cuda",  # use python3-9 when running on CPU
        description="Environment to run Llama 2 7B from Huggingface",
    ),
)
```

Package and upload the environment files.


```python
import shutil

training_environment_archive = shutil.make_archive(
    environment_dir, "zip", ".", environment_dir
)
api.environment_revisions_file_upload(
    project_name=PROJECT_NAME,
    environment_name=ENVIRONMENT_NAME,
    file=training_environment_archive,
)
```

## 3. Creating a deployment for the LLaMa 2 7B model

Now that we have created our code environment in UbiOps, it is time to write the actual code to run the `Llama-2-7B-HF` and push it to UbiOps.

As you can see we're uploading a `deployment.py` file with a `Deployment` class and two methods:
- `__init__` will run when the deployment starts up and can be used to load models, data, artifacts and other requirements for inference.
- `request()` will run every time a call is made to the model REST API endpoint and includes all the logic for processing data.

Separating the logic between the two methods will ensure fast model response times. We will load the model from Huggingface
in the `__init__` method, and code that needs to be ran when a call is made to the deployment in the `request()` method.
This way the model only needs to be loaded in when the deployment starts up. 


```python
deployment_code_dir = "deployment_code"
```


```python
!mkdir {deployment_code_dir}
```


```python
%%writefile {deployment_code_dir}/deployment.py
# Code to load from huggingface
"""
The file containing the deployment code needs to be called 'deployment.py' and should contain a 'Deployment'
class a 'request' method.
"""

import os
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
import torch
import shutil
from huggingface_hub import login
import os
import ubiops

class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. Any code inside this method will execute when the deployment starts up.
        It can for example be used for loading modules that have to be stored in memory or setting up connections.
        """


        PROJECT_NAME = context['project']

        self.REPETITION_PENALTY = float(os.environ.get('REPETITION_PENALTY', 1.15))
        self.MAX_RESPONSE_LENGTH  = float(os.environ.get('MAX_RESPONSE_LENGTH', 128))

        # Retrieve HF token
        model_hf_name = os.environ['model_id']
        token = os.environ["HF_TOKEN"]

        login(token=token)

        self.tokenizer = AutoTokenizer.from_pretrained(model_hf_name, use_auth_token= token)
        self.model = LlamaForCausalLM.from_pretrained(model_hf_name, use_auth_token= token)

        print(f"Model {model_hf_name} loaded")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        # Load model to GPU if available
        self.model.to(self.device)

        # Set model config
        self.generation_config = GenerationConfig(repetition_penalty=self.REPETITION_PENALTY)

        print("Initialising deployment")


    def request(self, data):

        """
        Method for deployment requests, called separately for each individual request.
        """
        prompt = data["prompt"]
        print(f"Running model on {self.device}")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs.to(self.device)

        # Generate
        with torch.no_grad():
          generate_ids = self.model.generate(inputs.input_ids, max_length=self.MAX_RESPONSE_LENGTH, generation_config=self.generation_config)

        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("full response")
        print(response)

        filtered_response = self.filter_prompt_from_response(prompt, response[0])
        return {"response": filtered_response}

    @staticmethod
    def filter_prompt_from_response(prompt, response_text):

      # Find the index where the prompt ends

      prompt_end_index = response_text.find(prompt) + len(prompt)

      # Get the generated response after the prompt

      filtered_response = response_text[prompt_end_index:].strip()

      return filtered_response
```

### Create a UbiOps deployment

Create a deployment. Here we define the in- and outputs of a model. We can create multiple versions.


```python
# Create the deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    input_type="structured",
    output_type="structured",
    input_fields=[{"name": "prompt", "data_type": "string"}],
    output_fields=[{"name": "response", "data_type": "string"}],
)

api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
```

### Create a deployment version

Now we will create a version of the deployment. For the version we need to define the name, the environment, the type 
of instance (CPU or GPU) as well the size of the instance.


```python
# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment=ENVIRONMENT_NAME,
    instance_type="180000mb_a100",  
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=600,  # = 10 minutes
    request_retention_mode="full",
)

api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=version_template
)
```

Package and upload the code




```python
# And now we zip our code (deployment package) and push it to the version

import shutil

deployment_code_archive = shutil.make_archive(
    deployment_code_dir, "zip", deployment_code_dir
)

upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file=deployment_code_archive,
)
print(upload_response)

# Check if the deployment is finished building. This can take a few minutes
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=upload_response.revision,
)
```

We can only send requests to our deployment version, after our environment has finished building. 

NOTE: Building the environment might take a while as we need to download and install all the packages and dependencies. We only need to build our environment once: next time that we spin up an instance of our deployment, we won't need to install all dependencies anymore. Toggle off `stream_logs` to not stream logs of the build process.

### Create an environment variable

Here we create an environment variable for the `model_id`, which is used to specify which model will be downloaded from Huggingface. If you want to use another version of Llama you can replace the value of `MODEL_ID` in the cell below, with the `model_id` of the model that you would like to use.


```python
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # You can change this parameter if you want to use a different model from Huggingface.
HF_TOKEN = "<HF TOKEN>"

api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=ubiops.EnvironmentVariableCreate(
        name="model_id", value=MODEL_ID, secret=False
    ),
)

api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=ubiops.EnvironmentVariableCreate(
        name="HF_TOKEN", value=HF_TOKEN, secret=False
    ),
)
```


```python
ubiops.utils.wait_for_deployment_version(
    api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    stream_logs=True,
)
```

# 4. Calling the LLaMa 2 7B deployment API endpoint

Our deployment is now ready to be requested! We can send requests to it via the `deployment-requests-create` or the `batch-deployment-requests-create` API endpoint. It is going to take some time before the request finishes. When our deployment first loads, a GPU node will need to spin up, and we will need to download the LLaMa 2 7B model from HuggingFace. Subsequent results to the deployment will be handled faster. We will use a batch request to kick off our instance. This way, we can stream the on-start logs, and monitor the progress of the request using the `ubiops.utils` library.


```python
data = {
    "prompt": "Tell me a fun fact about bears",
}

api.deployment_requests_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=data
).result
```

So that's it! You now have your own on-demand, scalable LLaMa 2 7B Chat HF model running in the cloud, with a REST API that you can reach from anywhere!
