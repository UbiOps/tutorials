# Deploy a Multi-Model OpenAI-Compatible Ollama Inference Server on UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ollama-multi-model-hosting/ollama-multi-model-hosting){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master//ollama-multi-model-hosting/ollama-multi-model-hosting){ .md-button .md-button--secondary }

In this tutorial, we will explain how to run multiple models supported by Ollama on UbiOps, including both chat completion models and embedding models within a single deployment. Developers distribute Ollama by publishing a custom `install.sh` [script](https://ollama.com/download). This script allows creating custom Docker images with Ollama by running `install.sh` in a `Dockerfile`. We will create a custom environment based on a UbiOps base environment (so that it [supports the requests format](https://ubiops.com/docs/deployments/docker-support/#supporting-request-format)) and deploy it using the [bring-your-own-image](https://ubiops.com/docs/environments/#bring-your-own-docker-image) feature. By the end, we will make both a chat completion and embedding requests to the multi-model Ollama server using Ubiops `requests` and OpenAI [python package](https://pypi.org/project/openai/).


## 1. Set up a connection with the UbiOps API client
First, we need to install the UbiOps Python Client Library to interface with UbiOps from Python:


```python
!pip install -qU ubiops openai
```

Now, we need to initialize all the necessary variables for the UbiOps deployment and set up the deployment directory, which we will later zip and upload to UbiOps.

To generate the API token you can follow this [guide](https://ubiops.com/docs/organizations/service-users/) (make sure you set up the right permissions).

Make sure you have access to the instance type `"16384 MB + 4 vCPU (Dedicated)"`, as for now only this instance supports Docker images.

Once you have your project name and the API token, paste them below in the following cell before running.


```python
API_TOKEN = "<UBIOPS_API_TOKEN>" # Used to create the deployments and pipeline, make sure this is in the format "Token token-code"
PROJECT_NAME = "<YOUR_PROJECT_NAME>" # Fill in your project name here
DEPLOYMENT_NAME = "ollama-multi-model-server"
ENVIRONMENT_NAME = "ollama-env"
DEPLOYMENT_VERSION = "v1"  
INSTANCE_TYPE = "16384 MB + 4 vCPU (Dedicated)"
API_HOST_URL="<API_HOST_URL>" # Standard UbiOps API URL is 'https://api.ubiops.com/v2.1', your URL may differ depending on your environment

print(f"Your new deployment will be called: {DEPLOYMENT_NAME}.")
```

Next, let's initialize the UbiOps client.


```python
import ubiops

configuration = ubiops.Configuration(host=API_HOST_URL)
configuration.api_key["Authorization"] = API_TOKEN

client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
api.service_status()
```

## 2. Creating a custom environment

For our multi-model Ollama deployment, we need to create a custom environment by uploading a Docker image to UbiOps. UbiOps has the [bring-your-own-image](https://ubiops.com/docs/environments/#bring-your-own-docker-image) feature, allowing you to upload custom Docker containers as environments while maintaining full compatibility with UbiOps request handling, scaling, and monitoring capabilities.

When you upload a Docker image as a [custom environment](https://ubiops.com/docs/environments/#create-a-custom-environment) in UbiOps, the platform treats it as any other environment that can be selected when creating deployment versions. The key requirement is that your Docker image must be compatible with UbiOps's request format and include the necessary agent implementation to handle incoming requests. 


### Prerequisites

- **Docker**: Install [Docker Engine](https://docs.docker.com/engine/install/) or [Docker Desktop](https://docs.docker.com/get-started/get-docker/) on your machine
- **UbiOps Base Image**: Access to UbiOps base environment images (contact your account manager or [support portal](https://support.ubiops.com) if unavailable). You will either receive access to a registry, or a single image tar file.


### 2.1 Pull or Load the base image with a UbiOps agent.

Pull a base image with a UbiOps agent (if you were granted access to a registry):


```python
!docker pull <registry>/ubiops-deployment-instance-ubuntu24.04-python3.13:v5.17.2
```

Load the base image with a UbiOps agent (if you have received a file):


```python
!docker load -i <FILE_DIRECTORY>
```

### 2.2 Create the dockerfile


```python
docker_file = """
FROM <registry>/ubiops-deployment-instance-ubuntu24.04-python3.13:v5.17.2
USER root
RUN apt-get update && \
    apt-get install --no-install-recommends -y git curl && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh

USER deployment

RUN pip install urllib3==1.26.19 jsonschema==3.2.0 django==5.1.4
RUN pip install ollama openai
"""

with open("Dockerfile", "w") as f:
    f.write(docker_file)
```

Now let's build the new image and save it as a `tar` archive.


```python
!docker build . -t ollama-ubiops
!docker save -o ollama-ubiops.tar ollama-ubiops
```

### 2.3 Creating an environment

We need to create an empty environment with `supports_request_format=True` in UbiOps that will serve as a container for our custom Docker image. This step establishes the environment definition in UbiOps, which we will then populate by uploading our custom Docker image.


```python
data = ubiops.EnvironmentCreate(
    name='ollama-env',
    description="Environment with an ollama server that supports requests format",
    supports_request_format=True
)
api.environments_create(PROJECT_NAME, data)
```

Now we can upload the image as a revision.


```python
api_response = api.environment_revisions_file_upload(
   PROJECT_NAME, 
   ENVIRONMENT_NAME, 
   file="ollama-ubiops.tar"
)
ubiops.utils.wait_for_environment(client, PROJECT_NAME, ENVIRONMENT_NAME)
api_response
```

## 3. Creating a UbiOps deployment
In this section, we will create the UbiOps deployment. 


### 3.1 Create UbiOps deployment
Now we can create the deployment, where we define the inputs and outputs of the model. Each deployment can have multiple versions. For each version, you can deploy different code, environments, instance types, etc.

The deployment will have `supports_request_format` enabled to allow autoscaling and monitoring of requests. We use the request endpoint to pass
payloads to the openai compatible chat completions endpoint. Therefore we will use input and output datatypes `plain`:

| Type   | Data Type |
|--------|-----------|
| Input  | Plain     |
| Output | Plain     |


```python
data = ubiops.DeploymentCreate(
    name = DEPLOYMENT_NAME,
    description = "Ollama multi-model deployment",
    supports_request_format=True,
    input_type = "plain",
    output_type = "plain"
)

deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data=data
)

print(deployment)
```

### 3.2 Create a deployment version
Next we create a version for the deployment. For the version we set the name, environment and size of the instance.

We also add labels to the deployment version to enable UbiOps's built-in model discovery system. These labels allow the platform to automatically expose your models through the /models endpoint.

Required labels:

- `openai-compatible: true` - Indicates this deployment can handle OpenAI-format requests

- `openai-model-names: model1;model2;model3` - Semicolon-separated list of model names served by this deployment


```python
labels = {
    "openai-compatible": "true",
    "openai-model-names": "smollm2;smollm;all-minilm-l6-v2;nomic-embed-text"
}


data = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment=ENVIRONMENT_NAME,
    instance_type_group_name=INSTANCE_TYPE,
    maximum_instances=1,
    minimum_instances=0,
    instance_processes=3,
    maximum_idle_time=900,
    labels = labels
)


deployment_version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=data,
)

print(deployment_version)
```

### 3.3 Creating a deployment directory

Let's create a deployment package directory, where we will add our [deployment package files](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/).


```python
import os

dir_name = "deployment_package"
# Create directory for the deployment if it does not exist
os.makedirs(dir_name, exist_ok=True)
```

### 3.4 Creating Deployment Code for UbiOps

We will now create the deployment code that will run on UbiOps. This involves creating a `deployment.py` file containing 
a `Deployment` class with two key methods:

- **`__init__` Method**  
  This method runs when the deployment starts. It can be used to load models, data artifacts, and other requirements for inference.

- **`request()` Method**  
  This method executes every time a call is made to the model's REST API endpoint. It contains the logic for processing incoming data.

We will configure [`instance_processes`](https://ubiops.com/docs/requests/request-concurrency/#request-concurrency-per-instance) to 3, 
allowing each deployment instance to handle 3 concurrent requests. Note that Ollama is not optimized to process a large number of requests simultaneously.

The Ollama server will be loaded as a background process within the `__init__` of the first process. A client will also be initialized in each process to proxy requests from all running processes to the Ollama server.

These environment variables will be set to configure Ollama's behavior:

- `OLLAMA_KEEP_ALIVE=-1`: will keep model always loaded in memory
- `OLLAMA_HOST=0.0.0.0:11434`: will serve Ollama on a public port. So, it can be also exposed through [port forwarding](https://ubiops.com/docs/deployments/deployment-versions/#opening-up-a-port-from-your-deployment-beta)
- `OLLAMA_EMBEDDING_MODELS` and `OLLAMA_CHAT_MODELS`: [environment variables](https://ubiops.com/docs/environment-variables/) configured to define which models the deployment should initialize and make available for inference (we will create these later in the notebook)

For a complete overview of the deployment code structure, refer to the [UbiOps documentation](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/).



```python
%%writefile {dir_name}/deployment.py
import subprocess
import os
import json
import time

from openai import OpenAI
import ollama


class PublicError(Exception):
    def __init__(self, public_error_message):
        super().__init__()
        self.public_error_message = public_error_message


class Deployment:

    def __init__(self, base_directory, context):
        print("Initializing deployment...")
        
        # Parse models from environment variables
        self.embedding_models = self._parse_models_env("OLLAMA_EMBEDDING_MODELS")
        self.chat_models = self._parse_models_env("OLLAMA_CHAT_MODELS")
        
        # Create model type mapping
        self.model_types = {}
        for model in self.embedding_models:
            self.model_types[model] = 'embedding'
        for model in self.chat_models:
            self.model_types[model] = 'chat'
        
        print(f"Loaded {len(self.embedding_models)} embedding models, {len(self.chat_models)} chat models")
        
        # Initialize OpenAI client
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key='ollama')
        self.environment_variables = {"OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_HOST": "0.0.0.0:11434"}
        
        # Initialize server and models (only in main process)
        if context.get('process_id') == 0:
            self._start_ollama_server()
            self._pull_all_models()
            print("Deployment ready")

    def request(self, data, context):
        """Process requests and route to appropriate model type."""
        input_data = json.loads(data) if isinstance(data, str) else data
        
        model_name = input_data.get('model')
        model_type = self.model_types[model_name]
        
        # Route request based on model type
        if model_type == 'embedding':
            response = self.client.embeddings.create(**input_data)
            return response.model_dump()
        else:  # chat model
            if input_data.get('stream', False):
                response = self.client.chat.completions.create(**input_data)
                full_response = []
                for chunk in response:
                    chunk_data = chunk.model_dump()
                    context['streaming_update'](json.dumps(chunk_data))
                    full_response.append(chunk_data)
                return full_response
            else:
                response = self.client.chat.completions.create(**input_data)
                return response.model_dump()

    def _parse_models_env(self, environment_variable):
        """Parse comma-separated model names from environment variable."""
        models_string = os.environ.get(environment_variable, "")
        return [model.strip() for model in models_string.split(',') if model.strip()]
    
    def _start_ollama_server(self):
        """Start the Ollama server process."""
        print("Starting Ollama server...")
        subprocess.Popen(['ollama', 'serve'], env=self.environment_variables | os.environ)
        time.sleep(3)
    
    def _pull_all_models(self):
        """Download all configured models."""
        for model_name in self.model_types.keys():
            print(f"Pulling model: {model_name}")
            ollama.pull(model_name)
```

We need to archive the deployment directory into a ZIP file before uploading to UbiOps. UbiOps requires all deployment packages to be uploaded as ZIP archives containing the deployment code and dependencies. For more details on the required package structure, see the [UbiOps deployment package documentation](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/).


```python
import shutil

# Archive the deployment directory
deployment_zip_path = shutil.make_archive(dir_name, 'zip', dir_name)
```

### 3.5 Upload a revision
We will now upload the deployment to UbiOps. In the background, This step will take some time, because UbiOps interprets
the environment files and builds a docker container out of it. You can check the UI for any progress.


```python
upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file=dir_name+".zip",
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

## 4. Creating environment variables

In order to be able to change the Ollama models deployed without modifying the deployment package, we can setup environment variables within the deployment and change their values based on our needs.
How it works:

`OLLAMA_EMBEDDING_MODELS` - Contains a comma-separated list of embedding models (e.g., "all-minilm:l6-v2,nomic-embed-text")

`OLLAMA_CHAT_MODELS` - Contains a comma-separated list of chat models (e.g., "smollm,smollm2"



```python
# Create OLLAMA_EMBEDDING_MODELS Environment Variable
embedding_data = ubiops.EnvironmentVariableCreate(
    name="OLLAMA_EMBEDDING_MODELS",
    value="all-minilm:l6-v2,nomic-embed-text",
    secret=False
)

response = api.deployment_environment_variables_create(
    PROJECT_NAME, 
    DEPLOYMENT_NAME,
    embedding_data
)

# Create OLLAMA_CHAT_MODELS Environment Variable
chat_data = ubiops.EnvironmentVariableCreate(
    name="OLLAMA_CHAT_MODELS",
    value="smollm2,smollm",
    secret=False
)

response = api.deployment_environment_variables_create(
    PROJECT_NAME, 
    DEPLOYMENT_NAME,
    chat_data
)
```

## 5. Making requests to the deployment
Our deployment is now live on UbiOps! Let's test it out by sending a bunch of requests to it. This request will be a simple prompt to the model, asking it to respond to a question. In case your deployment still needs to scale, it may take some time before your first request is picked up. You can check the logs of your deployment version to see if the Ollama server is ready to accept requests.

### 5.1 Send a request
Let's first create the request template and write some questions that we can send to the deployment.


```python
import copy
import json

request_template = {
    "messages": [
        {
            "content": "You are a helpful assistant.",
            "role": "system"
        },
        {
            "content": "{question}",
            "role": "user"
        }
    ],
    "model": "smollm2", 
    "stream": False
}

questions = [
    "What is the weather like today?",
    "How do I cook pasta?",
    "Can you explain quantum physics?",
    "What is the capital of France?",
    "How do I learn Python?"
]

requests_data = []
for question in questions:
    filled_request = copy.deepcopy(request_template)
    filled_request['messages'][1]['content'] = question
    requests_data.append(filled_request)

# Print the resulting requests
print(json.dumps(requests_data, indent=2))
```

Now let's create a request and print the results.


```python
print(api.deployment_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=requests_data[2]
))
```

Let's also try a different `chat-completion` model. For this we can just change the model name in the request template.


```python
request_template = {
    "messages": [
        {
            "content": "You are a helpful assistant.",
            "role": "system"
        },
        {
            "content": "What is the meaning of life?",
            "role": "user"
        }
    ],
    "model": "smollm", 
    "stream": False
}

print(api.deployment_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=request_template
))
```

### 5.2 Send a batch of requests
This section sends a batch of requests. It allows you to observe how Ollama fetches and processes multiple requests simultaneously.


```python
send_plain_batch = [json.dumps(item) for item in requests_data]

requests = api.batch_deployment_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME, 
    data=send_plain_batch, 
    timeout=3600
)
```

We can go over to the UI and inspect how are the `requests` being handled.

### 5.3 Sending embedding requests

Now let's test the embedding functionality. Embedding requests convert text into numerical vectors that capture semantic meaning, commonly used for search and RAG applications. We specify an embedding model and text input (instead of messages), and the result will be a vector representing the text's semantic meaning.


```python
embedding_request = {
    "model": "all-minilm:l6-v2",
    "input": "Can you tell me more about openai in exactly 50 words"
}

print(api.deployment_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=embedding_request
))

```

### 5.4 Sending a request with streaming output

For this request, we will add the key `stream: true` to the input, enabling streaming responses


```python
request_data = {
    "messages": [
        {
            "content": "You are a helpful assistant.",
            "role": "system"
        },
        {
            "content": "How is the weather?",
            "role": "user"
        }
    ],
    "model": "smollm2",
    "stream": True
}

# Create a streaming deployment request
for item in ubiops.utils.stream_deployment_request(
        client=api.api_client,
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        version=DEPLOYMENT_VERSION,
        data=request_data,
        timeout=3600,
        full_response=False,
):
    item_dict = json.loads(item)
    if item_dict.get("choices"):
        print(item_dict["choices"][0]["delta"]["content"], end="")
```

### 5.5 Sending requests to the OpenAI Endpoint
We can also connect to this deployment with the UbiOps OpenAI endpoint. 

First, let's initialize the client.


```python
from openai import OpenAI

client = OpenAI(
    api_key=API_TOKEN.lstrip("Token "),  
    base_url=f"{API_HOST_URL}/projects/{PROJECT_NAME}/openai-compatible/v1/"
)
```

Now we can create the request and print the result.


```python
stream_var = False

print(client.chat.completions.create(
    model=f"ubiops-deployment/{DEPLOYMENT_NAME}/{DEPLOYMENT_VERSION}/smollm",
    messages=[{"role": "user", "content": "Can you tell me more about openai in exactly two lines"}],
    stream=stream_var
))
```

Let's also create an embedding request.


```python
print(client.embeddings.create(
    model=f"ubiops-deployment/{DEPLOYMENT_NAME}/{DEPLOYMENT_VERSION}/all-minilm:l6-v2",
    input="Hello!"
))
```

We can also list the models that are available for your API token using the /models endpoint.


```python
models_list = client.models.list()

print(models_list)
```

## 6. Cleanup
At last, let's close our connection to UbiOps


```python
client.close()
```

We have set up a deployment that hosts a multi-model Ollama server. This tutorial just serves as an example. Feel free to reach out to
our [support portal](https://www.support.ubiops.com) if you want to discuss your set-up in more detail.
