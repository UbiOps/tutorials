# Deploy an Open-AI compatible Ollama inference server on UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ollama-tutorial/ollama-tutorial){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master//ollama-tutorial/ollama-tutorial){ .md-button .md-button--secondary }

In this tutorial, we will explain how to run LLMs supported by Ollama on UbiOps. Developers distribute Ollama by publishing a custom `install.sh` [script](https://ollama.com/download). This script allows creating custom Docker images with Ollama by running `install.sh` in a `Dockerfile`. We will create a custom environment based on a UbiOps base environment (so that it [supports the requests format](https://ubiops.com/docs/deployments/docker-support/#supporting-request-format)) and deploy it using the [bring-your-own-image](https://ubiops.com/docs/environments/#bring-your-own-docker-image) feature. Finally, we will make requests to Ollama server using Ubiops `requests` and OpenAI [python package](https://pypi.org/project/openai/).

Once you've completed this tutorial, you'll be able to successfully run an OpenAI-compatible Ollama inference server on UbiOps. If you're looking to take things a step further and host multiple Ollama models within a single deployment, be sure to check out our [Multi-Model Ollama Hosting](https://ubiops.com/docs/ubiops_tutorials/ollama-multi-model-hosting/ollama-multi-model-hosting/) tutorial.


## 1. Set up a connection with the UbiOps API client
First, we need to install the UbiOps Python Client Library to interface with UbiOps from Python:


```python
!pip install -qU ubiops openai
```

Now, we need to initialize all the necessary variables for the UbiOps deployment and set up the deployment directory, which we will later zip and upload to UbiOps.

To generate the API token you can follow this [guide](https://ubiops.com/docs/organizations/service-users/).

Make sure you have access to the `(Dedicated)` instance types, as for now only those instances support [bring-your-own-image](https://ubiops.com/docs/environments/#bring-your-own-docker-image) feature.

Once you have your project name and the API token, paste them below in the corresponding variable before running.


```python
API_TOKEN = "<INSERT API TOKEN WITH PROJECT EDITOR RIGHTS>"
PROJECT_NAME = "<INSERT YOUR PROJECT NAME>"
API_HOST_URL = "<INSERT YOUR HOST API URL>" # Standard UbiOps API URL is 'https://api.ubiops.com/v2.1', your URL may differ depending on your environment

DEPLOYMENT_NAME = "ollama-server"
ENVIRONMENT_NAME = "ollama-environment"
DEPLOYMENT_VERSION = "v1" # Choose a name for the version.
INSTANCE_TYPE = "16384 MB + 4 vCPU (Dedicated)"

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

For our Ollama deployment, we need to create a custom environment by uploading a Docker image to UbiOps. UbiOps has the [bring-your-own-image](https://ubiops.com/docs/environments/#bring-your-own-docker-image) feature, allowing you to upload custom Docker containers as environments.

When you upload a Docker image as a [custom environment](https://ubiops.com/docs/environments/#create-a-custom-environment) in UbiOps, the platform treats it as any other environment that can be selected when creating deployment versions. The image can either be set up to support the standard UbiOps [request format](https://ubiops.com/docs/deployments/docker-support/#supporting-request-format) , or it can be a non-request format (service) deployment.

For our goal of serving an Ollama model, we want to use the standard UbiOps [request format](https://ubiops.com/docs/deployments/docker-support/#supporting-request-format). This allows us to leverage more of of the platforms features, including automatic scaling based on incoming requests, detailed logging, and the ability to connect our model to other services using UbiOps Pipelines. To use the request format functionality with a custom Docker image, it is a key requirement that our image is based on an official UbiOps deployment image. This is because the base image contains the necessary agent that handles the communication with the UbiOps platform.

To achieve this and make our Docker image compatible, we will need the following prerequisites:


### Prerequisites

- **Docker**: Install [Docker Engine](https://docs.docker.com/engine/install/) or [Docker Desktop](https://docs.docker.com/get-started/get-docker/) on your machine
- **UbiOps Base Image**: Access to UbiOps base environment images (contact your account manager or [support portal](https://support.ubiops.com) if unavailable). You will either receive access to a registry, or a single image tar file. It is important to use the base image provided by UbiOps because it includes an agent implementation. This agent is what handles the UbiOps request format and starts your deployment code when a request arrives.

### 2.1 Pull or Load the base image.

If you have received access to a UbiOps registry, the image can be retrieved with the following command:


```python
!docker pull <registry>/ubiops-deployment-instance-ubuntu24.04-python3.13:v5.17.2
```

If you have received an image tar file, you can load the image with the following command::


```python
!docker load -i <PATH_TO_TAR_FILE>
```

### 2.2 Create the dockerfile

Create a `Dockerfile` that installs Ollama and the OpenAI client on top of the base image.


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

Build the new image and save it as a `tar` archive.


```python
!docker build . -t ollama-ubiops
!docker save -o ollama-ubiops.tar ollama-ubiops
```

### 2.3 Creating a UbiOps environment

First, we create an empty environment (without a base environment) in UbiOps with `supports_request_format=True` this reserves a spot for our Docker image.


```python
data = ubiops.EnvironmentCreate(
    name=ENVIRONMENT_NAME,
    description="Environment with an ollama server that supports requests format",
    supports_request_format=True,
)
api.environments_create(PROJECT_NAME, data)
```

Now we can upload the image as a revision.


```python
api_response = api.environment_revisions_file_upload(
   PROJECT_NAME, 
   ENVIRONMENT_NAME, 
   file="./ollama-ubiops.tar"
)
ubiops.utils.wait_for_environment(client, PROJECT_NAME, ENVIRONMENT_NAME)
api_response
```

## 3. Creating a UbiOps deployment
In this section, we will create the UbiOps deployment. 


### 3.1 Create UbiOps deployment
Now we can create the deployment, where we define the inputs and outputs of the model. Each deployment can have multiple versions. For each version, you can deploy different code, environments, instance types, etc. 

You can learn more about deployments on UbiOps [here](https://ubiops.com/docs/deployments/),

The deployment will have `supports_request_format` enabled to allow autoscaling and monitoring of requests. We use the request endpoint to pass
payloads to the [openai compatible](https://ubiops.com/docs/requests/openai/) chat completions and embeddings endpoint. Therefore we will use input and output datatypes `plain`:

| Type   | Data Type |
|--------|-----------|
| Input  | Plain     |
| Output | Plain     |


```python
data = ubiops.DeploymentCreate(
    name = DEPLOYMENT_NAME,
    description = "Ollama deployment",
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

We also add labels to the deployment version to enable UbiOps's built-in model discovery system. These labels allow the platform to automatically expose your models through the `/models` endpoint.

Required labels:

- `openai-compatible: true` - Indicates this deployment can handle OpenAI-format requests

- `openai-model-names: smollm` - Indicates the model that is used


```python
labels = {
    "openai-compatible": "true",
    "openai-model-names": "smollm2"
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
```

### 3.3 Creating a deployment directory

Let's create a deployment package directory, where we will add our [deployment package files](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/).


```python
import os

dir_name = "deployment_package"
# Create directory for the deployment if it does not exist
os.makedirs(dir_name, exist_ok=True)
```

### 3.5 Creating Deployment Code for UbiOps

We will now create the deployment code that will run on UbiOps. This involves creating a `deployment.py` file containing 
a `Deployment` class with two key methods:

- **`__init__` Method**  
  This method runs when the deployment starts. It can be used to load models, data artifacts, and other requirements for inference.

- **`request()` Method**  
  This method executes every time a call is made to the model's REST API endpoint. It contains the logic for processing incoming data.

We will configure [`instance_processes`](https://ubiops.com/docs/requests/request-concurrency/#request-concurrency-per-instance) to 3, 
allowing each deployment instance to handle 3 concurrent requests. The Ollama server will be loaded as a background process within the `__init__` 
of the first process. A client will also be initialized in each process to proxy requests from all running processes to the Ollama.

These environment variables will be set to optimize Ollama’s behavior:
- `OLLAMA_KEEP_ALIVE=-1`: will keep model always loaded in memory.
- `OLLAMA_HOST=0.0.0.0:11434`: will serve Ollama on a public port. So, it can be also exposed through [port forwarding](https://ubiops.com/docs/deployments/deployment-versions/#opening-up-a-port-from-your-deployment-beta).

For a complete overview of the deployment code structure, refer to the [UbiOps documentation](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/).



```python
%%writefile {dir_name}/deployment.py
import subprocess
import os
import logging
import json
import time

from openai import OpenAI, BadRequestError

logging.basicConfig(level=logging.INFO)

import ollama

class PublicError(Exception):
    def __init__(self, public_error_message):
        super().__init__()
        self.public_error_message = public_error_message
        

class Deployment:

    def __init__(self, base_directory, context):
        print("Initializing deployment")

        self.model_name = os.environ.get("MODEL_NAME", "smollm2")
        
        # In every process, initiate a client
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.envs = {"OLLAMA_KEEP_ALIVE":"-1", "OLLAMA_HOST": "0.0.0.0:11434"}

        if context["process_id"] == 0:
            print("Initializing Ollama server...")

            #Serve ollama as a background process
            subprocess.Popen(["ollama", "serve"], env=self.envs | os.environ)
            time.sleep(5) # wait for ollama to be served
            ollama.pull(self.model_name)
            self.poll_health_endpoint()

    def request(self, data, context):
        """
        Processes incoming requests using the OpenAI-compatible API.
        """
        print("Processing request")
        input_data = json.loads(data)
        stream_boolean = input_data.get("stream", False)  # Default to streaming
        input_data["model"] = self.model_name
        if stream_boolean:
            input_data["stream_options"] = {"include_usage": True}
        try:
            response = self.client.chat.completions.create(**input_data)
        except BadRequestError as e:
            raise PublicError(str(e))

        if stream_boolean:
            streaming_callback = context["streaming_update"]
            full_response = []
            for partial_response in response:
                chunk_dump = partial_response.model_dump()
                streaming_callback(json.dumps(chunk_dump))
                full_response.append(chunk_dump)
            return full_response
        return response.model_dump()

    def poll_health_endpoint(self):
        """
        Curls the Ollama server to ensure it's initialized.
        """
        print("Waiting for Ollama server to be ready...")

        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You warmed up yet?"}
            ],
            stream = False
            )
            print(f"{self.model_name}'s first response: \n {response}")

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            raise  # Exit on error and raise exception

                

```

We need to archive the deployment directory into a ZIP file before uploading to UbiOps. UbiOps requires all deployment packages to be uploaded as ZIP archives containing the deployment code and dependencies. For more details on the required package structure, see the [UbiOps deployment package documentation](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/).


```python
import shutil

# Archive the deployment directory
deployment_zip_path = shutil.make_archive(dir_name, 'zip', dir_name)
```

### 3.5 Upload a revision
We will now upload the deployment to UbiOps. In the background, this step will take some time, because UbiOps interprets
the environment files and builds a docker container out of it. You can check the UI for any progress. 

If you want to find out more about how a container is build with UbiOps you can check out [this page](https://ubiops.com/docs/deployments/revisions-building/).


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

## 4. Making requests to the deployment
Our deployment is now live on UbiOps! Let's test it out by sending a bunch of [requests](https://ubiops.com/docs/requests/) to it. This request will be a simple prompt to the model, asking it to respond to a question. In case your deployment still needs to scale, it may take some time before your first request is picked up. You can check the logs of your deployment version to see if the Ollama server is ready to accept requests.

### 4.1 Send a single request
Let's first create the request template and write a list of questions that we can choose from and send to the deployment.


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

Now let's send the request with a question and print the results.


```python
print(api.deployment_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=requests_data[1]
))
```

### 4.2 Send a batch of requests

This cell sends a batch of requests. It allows you to observe how Ollama fetches and processes multiple requests simultaneously. 


```python
send_batch = [json.dumps(item) for item in requests_data]

requests = api.batch_deployment_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME, 
    data=send_batch, 
    timeout=3600
)
```

### 4.3 Sending a request with streaming output

For this request, we will add the key `stream: true` to the input, enabling [streaming responses](https://ubiops.com/docs/requests/#streaming-requests).


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

### 4.4 Sending requests to the OpenAI Endpoint
We can also connect to this deployment with the UbiOps OpenAI endpoint.
Let's send the same messages, but through the OpenAI endpoint!

First, let's initialize the OpenAI client.


```python
from openai import OpenAI

client = OpenAI(
    api_key=API_TOKEN.lstrip("Token "),
    base_url=f"https://api.ubiops.com/v2.1/projects/{PROJECT_NAME}/openai-compatible/v1/"
)
```

Now we can create the request and print the result.


```python
stream_var = False
response = client.chat.completions.create(
    model=f"ubiops-deployment/{DEPLOYMENT_NAME}/{DEPLOYMENT_VERSION}/smollm2",
    messages=[{"role": "user", "content": "Can you tell me more about openai in exactly two lines"}],
    stream=stream_var
)
```

## 5. Listing the models.

Time to put those deployment version labels to work! We'll now use them to expose the full list of models through the /models endpoint.


```python
models_list = client.models.list()

print(models_list)
```

## 6. Cleanup
At last, let's close our connection to UbiOps


```python
client.close()
```

We have set up a deployment that hosts an Ollama server. This tutorial just serves as an example. Feel free to reach out to
our [support portal](https://www.support.ubiops.com) if you want to discuss your set-up in more detail.
