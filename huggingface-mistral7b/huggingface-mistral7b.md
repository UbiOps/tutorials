# Deploying Mistral 7B to UbiOps with a development set-up

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/huggingface-mistral7b/huggingface-mistral7b/mistral_7b_deployment.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/huggingface-mistral7b/huggingface-mistral7b/mistral_7b_deployment.ipynb){ .md-button .md-button--secondary }

This notebook will help you create a cloud-based inference API endpoint for the [Mistral-2-7B-Instruct model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
, using UbiOps. The Mistral version we will be using is already pretrained and will be 
loaded from the Huggingface [Mistral AI](https://huggingface.co/mistralai) library. The model has been developed by Mistral AI.

[Mistral 7B](https://arxiv.org/abs/2310.06825) is a collection of language model engineered for superior performance and 
efficiency. Mistral AI claims that the Mistral 7B outperforms Llama 2 13B across all evaluated benchmarks. The model 
deployed in this tutorial is a fine-tuned version of the Mistral 7B.

We will set up the deployment to handle various input configurations and master prompts. You can use this test setup to 
experiment with different inputs and configurations.

In this notebook we will walk you through:

1. Connecting with the UbiOps API client
2. Creating a code environment for our deployment
3. Creating a deployment with a test set-up for the Mistral 7B model
4. Calling the Mistral 7B deployment API endpoint

Mistral-7B is a text-to-text model. Therefore we will make a deployment which takes a text prompt as an input, and returns
a response. We will also add the `system_prompt` and [`config`](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig)
to the input of the deployment, so we can experiment with different inputs to see how that changes the response of the model. 
Note that Mistral is behind a gated repository - in order for you to download the model you will also need to a Huggingface 
token that has sufficient permissions to download Mistral.

Next to the response, the deployment will also return the used `input` (which consists of the `system_prompt` and the 
`prompt`) and the `used_config`.

When no additional `system_prompt` or `config` are provided, the deployment will use pre-set default values, which
you can find in the `__init__` statement of the deployment.

|Deployment input & output variables| **Variable name** |**Data type** |
|--------------------|--------------|--------------|
| **Input fields**   | prompt | string |
|                    | system_prompt | string |
|                    | config | dictionary|
| **Output fields**  | response | string |
|                    | input        | string |
|                    | used_config  | dictionary |

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

API_TOKEN = "<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>"  # Make sure this is in the format "Token token-code"
PROJECT_NAME = "<INSERT PROJECT NAME IN YOUR ACCOUNT>"

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"  # You can change this parameter if you want to use a different model from Huggingface.
HF_TOKEN = "<ENTER YOUR HF TOKEN WITH ACCESS TO MISTRAL REPO HERE>"

DEPLOYMENT_NAME = f"mistral-7b-{datetime.now().date()}"
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
ENVIRONMENT_NAME = "mistral-7b-environment"
```


```python
%mkdir {environment_dir}
```

We first write a `requirements.txt` file, which contains the Python packages that we will use in our deployment code


```python
%%writefile {environment_dir}/requirements.txt
# This file contains package requirements for the environment
# installed via PIP.
numpy==1.26.3
torch==2.0.1+cu118
transformers==4.37.0
accelerate==0.26.1
bitsandbytes==0.42.0
huggingface_hub==0.23.0
```

Next we add a `ubiops.yaml` to set a remote pip index. This ensures that we install a CUDA-compatible version of PyTorch. 
CUDA allows models to be loaded and to run GPUs.


```python
%%writefile {environment_dir}/ubiops.yaml
environment_variables:
- PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118
```

Now we create a UbiOps [`environment`](https://ubiops.com/docs/environments/#environments). We select python3.11 with CUDA 
pre-installed as the `base environment` if we want to run on GPUs. If we run on CPUs, then we use `python3-11`.

Our additional dependencies are installed on top of this base environment, to create our new `custom_environment` 
called `mistral-7b-environment`.


```python
api_response = api.environments_create(
    project_name=PROJECT_NAME,
    data=ubiops.EnvironmentCreate(
        name=ENVIRONMENT_NAME,
        # display_name=ENVIRONMENT_NAME,
        base_environment="python3-11-cuda",  # use python3-11 when running on CPU
        description="Environment to run Mistral 7B from Huggingface",
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

## 3. Creating a deployment for the Mistral 7B model

Now that we have created our code environment in UbiOps, it is time to write the actual code to run the Mistral-7B-Instruct-v0.2
model and push it to UbiOps.

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
import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig, pipeline
from huggingface_hub import login


class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. Any code inside this method will execute when the deployment starts up.
        It can for example be used for loading modules that have to be stored in memory or setting up connections.
        """

        print("Initialising deployment")

        model_id = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
        hf_token = os.environ["HF_TOKEN"]

        
        print(f"Model set as: {model_id}")

        login(token=hf_token)

        print("Login succesful")

        gpu_available = torch.cuda.is_available()
        print("Loading device")
        self.device = torch.device("cuda") if gpu_available else torch.device("cpu")
        print("Device loaded in")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        print("Downloading model")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                          quantization_config = bnb_config,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map="auto"
                                                          )
        
        print("Downloading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)


        self.pipe = pipeline(
            os.environ.get("PIPELINE_TASK", "text-generation"),
            model=model_id,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )

        self.base_prompt = (
            "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
        )

        self.default_config = {
            'do_sample': True,
            'max_new_tokens': 512,
            'eos_token_id': self.tokenizer.eos_token_id,
            'temperature': 0.3
        }    

        self.system_prompt = "You are a friendly chatbot who always responds in the style of a pirate"

    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        """

        prompt = data["prompt"]

        # Update the system prompt if user added a system prompt
        if data["system_prompt"]:
            system_prompt = data["system_prompt"]
        else:
            system_prompt = self.system_prompt

        config = self.default_config.copy()

        # Update config dict if user added a config dict
        if data["config"]:
            config.update(data["config"])

        #Create full prompt
        input = self.base_prompt.format(
            system_prompt=system_prompt, user_prompt=data["prompt"]
        )

        model_inputs = self.tokenizer(input, return_tensors="pt").to(self.device)

        # Here we set the GenrerationConfig to parameteriz the generate method
        generation_config = self.default_config.copy()
        
        #Update config dict if user added a config dict
        if data["config"]:
            generation_config.update(data["config"])
        

        print("Generating output")

        # Generate text
        sequences = self.pipe(
            input,
            **config
        )
                                            
        response = sequences[0]["generated_text"]

        # Here we set our output parameters in the form of a json
        return {"response": response,
                "used_config": config,
                "input":input}

```

### Create a UbiOps deployment

Create a deployment. Here we define the in- and outputs of a model. We can create different deployment versions.

Note that we added a default `system_prompt` & `config` field to the input.

```python
# Create the deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    input_type="structured",
    output_type="structured",
    input_fields=[
        {"name": "prompt", "data_type": "string"},
        {"name": "system_prompt", "data_type": "string"},
        {"name": "config", "data_type": "dict"},
    ],
    output_fields=[
        {"name": "response", "data_type": "string"},
        {"name": "input", "data_type": "string"},
        {"name": "used_config", "data_type": "dict"},
    ],
)

api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
```

### Create a deployment version

Now we will create a version of the deployment. For the version we need to define the name, the environment, the type of instance (CPU or GPU) as well the size of the instance.


```python
# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment=ENVIRONMENT_NAME,
    instance_type_group_name="16384 MB + 4 vCPU + NVIDIA Tesla T4",
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

Here we create an environment variable for the `model_id` and your, which is used to specify which model will be downloaded from Huggingface. If you want to use another version of Mistral you can replace the value of `MODEL_ID` in the cell below, with the `model_id` of the model that you would like to use.

Here we create two environment variables, one for the `model_id` and your, which is used to specify which model will be 
downloaded from Huggingface. And one for your Huggingface token, which you need to download the model from Huggingface.
If you want to use another version of Mistral you can replace the value of `MODEL_ID` in the cell below, with the 
`model_id` of the model that you would like to use.



```python

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
    data=ubiops.EnvironmentVariableCreate(name="HF_TOKEN", value=MODEL_ID, secret=True),
)
```

## 4. Calling the Mistral 7B deployment API endpoint

Our deployment is now ready to be requested! We can send requests to it via the `deployment-requests-create` or the `batch-deployment-requests-create` API endpoint. During this step a node will be spun up, and the model will be downloaded
from Huggingface. Hence why this step can take a while. You can monitor the progress of the process in the 
[logs](https://ubiops.com/docs/monitoring/logging/). Subsequent results to the deployment will be handled faster. We 
will use a batch request to kick off our instance.

##### Make a request using the default `system_prompt` and `config`.


```python
data = {"prompt": "tell me a joke", "system_prompt": "", "config": {}}

api.deployment_requests_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=data, timeout=3600
).result
```

##### Make a request using other values for the `system_prompt` and `config`.

For this request we will instruct the LLM to translate English texts into the style of Shakespearean. We will let the model
be more creative with generating sequences by lowering the `temperature` parameter. The text used for this example is shown
in the cell below:


```python
text = "In the village of Willowbrook lived a girl named Amelia, known for her kindness and curiosity. One autumn day, she ventured into the forest and stumbled upon an old cottage filled with dusty tomes of magic. Amelia delved into the ancient spells, discovering her own hidden powers. As winter approached, a darkness loomed over the village. Determined to protect her home, Amelia confronted the source of the darkness deep in the forest. With courage and magic, she banished the shadows and restored peace to Willowbrook.Emerging triumphant, Amelia returned home, her spirit ablaze with newfound strength. From that day on, she was known as the brave sorceress who saved Willowbrook, a legend of magic and courage that echoed through the ages."
```


```python
data = {
    "prompt": text,
    "system_prompt": "You are a friendly chatbot that translates texts into the style of Shakespearean.",
    "config": {
        "do_sample": True,
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.5,
    },
}

api.deployment_requests_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=data, timeout=3600
).result
```

So that's it! You now have your own on-demand, scalable Mistral-7B-Instruct-v0.2 model running in the cloud, with a REST API that you can reach from anywhere!
