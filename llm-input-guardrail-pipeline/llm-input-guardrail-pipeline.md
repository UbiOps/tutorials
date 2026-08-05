# Implement custom input gaurdrails on UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/llm-input-guardrail-pipeline/llm-input-guardrail-pipeline/llm-input-guardrail-pipeline.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/llm-input-guardrail-pipeline/llm-input-guardrail-pipeline/llm-input-guardrail-pipeline.ipynb){ .md-button .md-button--secondary }

This notebook shows an example on how to implement input custom guardrails into a UbiOps pipeline. Input guardrails are mechanisms 
designed to filter and validate user input before it reaches the LLM. The input request can then either be blocked, or adjusted,
to steer the LLM response into a certain direction.

Different input guardrail mechanisms exist. You can use a high-throughput LLm to classify a response, or simply use regex.
As an example, we will guide you through a simple regex guardrail example.

We will implement a pipeline that connects two deployments. One deployment applies the regex filter. The second deployment
proxies a request to an LLM. The return of the LLM will be streamed.

The pipeline can be called with OpenAI-compatible input bodies.  In the next release of UbiOps, the pipeline will be exposed
via an [OpenAI compatible chat/completions endpoint](https://ubiops.com/docs/requests/openai/#openai-compatible-requests).

This solution can be set up in your UbiOps environment in four steps:
1) Establish a connection with your UbiOps environment
2) Create the deployment for the input guardrailing
3) Create the deployment for the proxy LLM, using connection strings
4) Create a pipeline that combines the two deployments created in step 2 and 3


## 1) Connecting with the UbiOps API client

To use the UbiOps API from our notebook, we need to install the UbiOps Python client library.


```python
%pip install --upgrade ubiops
```

To set up a connection with the UbiOps platform API we need the name of your UbiOps project and an API token with `project-editor` permissions.

Once you have your project name and API token, paste them below in the following cell before running.


```python
import json

import ubiops

API_TOKEN = "<UBIOPS_API_TOKEN>"  # Used to create the deployments and pipeline, make sure this is in the format "Token token-code"
API_HOST_URL = "<API_HOST_URL>"  # Standard UbiOps API URL is 'https://api.ubiops.com/v2.1', your URL may differ depending on your environment
PROJECT_NAME = "<PROJECT_NAME>"  # Fill in your project name here

configuration = ubiops.Configuration()
configuration.api_key["Authorization"] = API_TOKEN
configuration.host = API_HOST_URL

api_client = ubiops.ApiClient(configuration)
api = ubiops.api.CoreApi(api_client)


print(api.service_status())
```

You will also need to be able to send requests to an LLM that accepts [OpenAI compatible chat completion requests](https://ubiops.com/docs/requests/openai/#openai-compatible-requests). You can use an external supplier, such as OpenAI or Mistral, or configure such an LLM on UbiOps yourself.


```python
BASE_URL = "<BASE_URL>" # The base URL of your LLM. If hosted on UbiOps, use f"{API_HOST_URL}/projects/{PROJECT_NAME}/openai-compatible/v1/"
MODEL_NAME = "<MODEL_NAME>" # The name of your LLM model. If hosted on UbiOps, use  f"ubiops-deployment/{DEPLOYMENT_NAME}//<the name of your model>"
API_KEY = "<API_KEY>" # Used to create requests within the proxy deployment. If hosted on UbiOps, use a valid API Token with atleast `deployment-request-user` permissions to request the deployment, but without the 'Token ' prefix
```

## Create the deployments for the pipeline

Now that we have established a connection with our UbiOps environment, we can start creating our deployment packages. Each
package will consist of two files:
- The `deployment.py`, which is where we will define the actual code to run the embedding model and LLM
- The `requirements.txt`, which will contain additional dependencies that our codes needs to run properly

These deployment packages will be zipped, and uploaded to UbiOps, after which we will add them to a pipeline. The pipeline
will consist out of two deployments:
- One deployment will host the embedding model
- One will host the LLM


```python
GUARDRAIL_DEPLOYMENT_NAME = "filter-apple-guardrail"
GUARDRAIL_DEPLOYMENT_PACKAGE_DIR = "input_guardrail_deployment_package"
PROXY_LLM_DEPLOYMENT_NAME = "proxy-llm"
PROXY_LLM_DEPLOYMENT_PACKAGE_DIR = "proxy_llm_deployment_package"
```

### 2) Create the Input guardrail deployment

This deployment adds a simple input guardrail before messages reach the main LLM. It checks if the user mentions the word "apple" and, if so, inserts a system message instructing them to talk about other fruits instead. It also validates that the input is properly formatted JSON with a "messages" list. If not, a public error is returned to the end-user.


```python
%mkdir {GUARDRAIL_DEPLOYMENT_PACKAGE_DIR}
```

First we create the `deployment.py`:


```python
%%writefile {GUARDRAIL_DEPLOYMENT_PACKAGE_DIR}/deployment.py
import re
import json

class Deployment:
    def __init__(self):
        self.guard = SimpleWordChecker()

    def request(self, data: str) -> str:
        # Load the OpenAI-Compatible input body
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            raise PublicError("Invalid JSON: Could not parse request body.")

        # Validate required structure
        if "messages" not in parsed or not isinstance(parsed["messages"], list):
            raise PublicError("Invalid input: 'messages' key must be present and must be a list.")

        updated = self.guard.check_for_apple(parsed)

        # Dump the response back as a string
        return json.dumps(updated)


class SimpleWordChecker:
    def check_for_apple(self, body):
        '''
        Checks if the last user message contains the word "apple", 
        if so, adds a new system message for the LLM.

        Returns the body of messages
        '''

        messages = body["messages"]
        for i in reversed(range(len(messages))):
            if messages[i].get("role") == "user":

                # Find the last user message and check for the forbidden word
                if re.search(r"\bapple\b", messages[i].get("content", ""), re.IGNORECASE):
                    messages.insert(i + 1, {
                        "role": "system",
                        "content": "The user used the word apple. Using the word apple is forbidden. Instruct the user to talk about other fruits instead.."
                    })
                # We only need to check the last user's message, so stop after this
                break

        return body

class PublicError(Exception):
    '''
    Raise a public error message to the user 
    which is visible from the request overview page in UbiOps.
    '''
    
    def __init__(self, public_error_message):
        super().__init__()
        self.public_error_message = public_error_message


```

#### Now we create the deployment 

For the deployment we will specify the in- and output for the model as type `plain`, to support OpenAI-compatible input:


```python
deployment_template = ubiops.DeploymentCreate(
    name=GUARDRAIL_DEPLOYMENT_NAME,
    description="An example deployment that checks if a user used the world apple and instructs the LLM to make the end-user" \
    "aware of this.",
    input_type="plain",
    output_type="plain",
    labels={"type": "input-guardrail"},
)

guardrail_deployment = api.deployments_create(
    project_name=PROJECT_NAME, data=deployment_template
)
print(guardrail_deployment)
```

#### And finally we create the version

Each deployment can have multiple versions. The version of a deployment defines the coding environment, instance type (CPU or GPU) 
& size, and other settings. We will set `minimum_instances` to warrant fast response time . The code is simple and will not consume a lot of resources. Therefore we select the smallest instance type group available.

⚠️ **Warning:** toggle `minimum_instances` to `0` after this tutorial to save up on resources


```python
version_template = ubiops.DeploymentVersionCreate(
    version="v1",
    environment="python3-13",
    instance_type_group_name="256 MB + 0.0625 vCPU",
    minimum_instances=1,
    maximum_instances=1,
    maximum_idle_time=10,
    instance_processes = 1,
    request_retention_mode="full",  # Input/output of requests will be stored.
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name=GUARDRAIL_DEPLOYMENT_NAME, data=version_template
)
print(version)
```

Then we zip the `deployment package` and upload it to UbiOps (this process can take between 5-10 minutes). 


```python
import shutil

shutil.make_archive(GUARDRAIL_DEPLOYMENT_PACKAGE_DIR, "zip", ".", GUARDRAIL_DEPLOYMENT_PACKAGE_DIR)

file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=GUARDRAIL_DEPLOYMENT_NAME,
    version="v1",
    file=f"{GUARDRAIL_DEPLOYMENT_PACKAGE_DIR}.zip",
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=GUARDRAIL_DEPLOYMENT_NAME,
    version="v1",
    revision_id=file_upload_result.revision,
)
```

### 3) Create the proxy LLM deployment

Next we will create the deployment that will proxy a request to an external LLM by hitting its `v1/chat/completions` endpoint. The deployment functions simply as a passthrough, although
you're free to add custom logic. The workflow for creating this deployment is similar to the workflow for creating the previous deployment, except we now add the `openai` python package to the environment, and use environment variables to specify the LLM to which we will apply requests.


```python
%mkdir {PROXY_LLM_DEPLOYMENT_PACKAGE_DIR}
```

Create the `deployment.py`:


```python
%%writefile {PROXY_LLM_DEPLOYMENT_PACKAGE_DIR}/deployment.py

import os
import json

from openai import OpenAI

class Deployment:
    def __init__(self, base_directory, context):
        print("Initializing OpenAI-compatible Deployment")

        try:
            self.base_url = os.environ["BASE_URL"]
            self.model_name = os.environ["MODEL_NAME"]
            self.api_key = os.environ["API_KEY"]  # You might want to add this or similar
        except KeyError as e:
            raise Exception(f"Missing required environment variable: {e}")

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.context = context

    def request(self, data, context):
        print("Processing request for OpenAI-compatible Deployment")

        try:
            input_data = json.loads(data)
        except (TypeError, ValueError):
            raise PublicError("Invalid JSON: Could not parse request body.")

        input_data["model"] = self.model_name

        # Optional: Include usage info for tracking tokens 
        is_streaming = input_data.get("stream", False)
        if is_streaming:
            input_data["stream_options"] = {"include_usage": True}

        try:
            print(f"Sending request to model {input_data['model']}")
            response = self.client.chat.completions.create(**input_data)
        except Exception as e:
            raise RuntimeError("Failed to call model") from e

        if is_streaming:
            streaming_callback = context["streaming_update"]
            full_response = []
            for partial_response in response:
                chunk_dump = partial_response.model_dump()
                streaming_callback(json.dumps(chunk_dump))
                full_response.append(chunk_dump)
            return json.dumps(full_response)
        else:
            full_response = response.model_dump()
            return json.dumps(full_response)


class PublicError(Exception):
    def __init__(self, public_error_message):
        super().__init__()
        self.public_error_message = public_error_message
    
```

Then the `requirements.txt`:


```python
%%writefile {PROXY_LLM_DEPLOYMENT_PACKAGE_DIR}/requirements.txt
openai
```

#### Create a deployment

Again, we will use input and output types `plain`


```python
llm_template = ubiops.DeploymentCreate(
    name=PROXY_LLM_DEPLOYMENT_NAME,
    description="A deployment that proxies requests to an OpenAI-compatible server",
    input_type="plain",
    output_type="plain",
    labels={"type": "llm-proxy"},
)

llm_deployment = api.deployments_create(project_name=PROJECT_NAME, data=llm_template)
print(llm_deployment)
```

And create a version for the deployment. We will use a slightly larger instance type to ensure that the llm proxy can
handle multiple requests concurrently:


```python
version_template = ubiops.DeploymentVersionCreate(
    version="v1",
    environment="python3-13",
    instance_type_group_name="512 MB + 0.125 vCPU",
    maximum_instances=1,
    minimum_instances=1,
    maximum_idle_time=10, 
    instance_processes=5,
    request_retention_mode="full",  # input/output of requests will be stored)
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name=PROXY_LLM_DEPLOYMENT_NAME, data=version_template
)
print(version)
```

Now we need to create environment variables that allow the proxy deployment to request an LLM.


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=PROXY_LLM_DEPLOYMENT_NAME,
    version="v1",
    data=ubiops.EnvironmentVariableCreate(name="BASE_URL", value=BASE_URL, secret=False),
)

api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=PROXY_LLM_DEPLOYMENT_NAME,
    version="v1",
    data=ubiops.EnvironmentVariableCreate(name="MODEL_NAME", value=MODEL_NAME, secret=False),
)

api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=PROXY_LLM_DEPLOYMENT_NAME,
    version="v1",
    data=ubiops.EnvironmentVariableCreate(name="API_KEY", value=API_KEY, secret=True),
)
```

Zip & upload the files to UbiOps (this process can take between 5-10 minutes).


```python
import shutil

shutil.make_archive(PROXY_LLM_DEPLOYMENT_PACKAGE_DIR, "zip", ".", PROXY_LLM_DEPLOYMENT_PACKAGE_DIR)

file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=PROXY_LLM_DEPLOYMENT_NAME,
    version="v1",
    file=f"{PROXY_LLM_DEPLOYMENT_PACKAGE_DIR}.zip",
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=PROXY_LLM_DEPLOYMENT_NAME,
    version="v1",
    revision_id=file_upload_result.revision,
)
```

## 4) Create a pipeline and pipeline version

Now we create a pipeline that orchestrates the workflow between the deployments above. When a request will be made to this pipeline
the first deployment will check the last user's prompt for forbidden words. Then it passes the messages through to the LLM to generate an answer.

For a pipeline you will have to define an input & output and create a version, as with a deployment. In addition to this we
will also need to define the objects (i.e, deployments) and how to orchestrate the workflow (i.e., how to attach each object
 to each other).

First we create the pipeline:


```python
PIPELINE_NAME = "guardrail-pipeline-demo"
PIPELINE_VERSION = "v1"
```


```python
pipeline_template = ubiops.PipelineCreate(
    name=PIPELINE_NAME,
    description="A pipeline that applies an input guardrail",
    input_type="plain",
    output_type="plain"
)

api.pipelines_create(project_name=PROJECT_NAME, data=pipeline_template)
```

Then we define the objects, and how to attach the objects together:


```python
# Define the two objects to be used in the pipeline

objects = [
    # input-guardrail
    {
        "name": GUARDRAIL_DEPLOYMENT_NAME,
        "reference_name": GUARDRAIL_DEPLOYMENT_NAME,
        "version": "v1",
    },
    # LLM-model
    {
        "name": PROXY_LLM_DEPLOYMENT_NAME, 
        "reference_name": PROXY_LLM_DEPLOYMENT_NAME, 
        "version": "v1"
     },
]

attachments = [
    # start --> input-guardrail
    {
        "destination_name": GUARDRAIL_DEPLOYMENT_NAME,
        "sources": [
            {
                "source_name": "pipeline_start",
                "mapping": [],
            }
        ],
    },
    # input-guardrail --> LLM
    {
        "destination_name": PROXY_LLM_DEPLOYMENT_NAME,
        "sources": [
            {
                "source_name": GUARDRAIL_DEPLOYMENT_NAME,
                "mapping": []
            }
        ],
    },
    # LLM --> pipeline end
    {
        "destination_name": "pipeline_end",
        "sources": [
            {
                "source_name": PROXY_LLM_DEPLOYMENT_NAME,
                "mapping": [],
            }
        ],
    },
]
```

And finally we create a version for this pipeline. Note that we are adding labels, so that the solutions will be returned
when using the `/models` [endpoint](https://ubiops.com/docs/requests/openai/#listing-available-models):


```python
pipeline_template = ubiops.PipelineVersionCreate(
    version=PIPELINE_VERSION,
    request_retention_mode="full",
    objects=objects,
    attachments=attachments,
    labels = {"openai-model-names":"llm-apple-input-guardrail", "openai-compatible":True}
)

api.pipeline_versions_create(
    project_name=PROJECT_NAME, pipeline_name=PIPELINE_NAME, data=pipeline_template
)
```

## And there you have it!

We have now set up input guardrails on UbiOps. If you want, you can use the code block below to create a request to your newly created pipeline.


```python
data ="""
{
    "messages": [
        {
            "content": "You are a helpful assistant.",
            "role": "system"
        },
        {
            "content": "I ate an apple!",
            "role": "user"
        }
    ],
    "stream": false
}
"""
```


```python
response = api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=PIPELINE_NAME,
    data=data
)

print(response)
```

The textual response of the LLM reads as


```python
import json 
json.loads(response.result)["choices"][0]["message"]["content"]
```

An example response would be:

"""I see you mentioned a certain fruit that starts with "A". Let's try something different. How about we talk about bananas, oranges, or grapes instead? Which one of those fruits do you like?"""

You can also initiate requests via the `openai` library:


```python
from openai import OpenAI

client = OpenAI(
    api_key= API_TOKEN[6:] if API_TOKEN.startswith("Token ") else API_TOKEN,
    base_url = f"{API_HOST_URL}/projects/{PROJECT_NAME}/openai-compatible/v1"

)
```

Then fetch all models available within your project:


```python
models = client.models.list()
print(models)
```


```python
response = client.chat.completions.create(
    model=MODEL_NAME,
    **json.loads(data)
)
print(response)
```

## 6. Cleanup
At last, let's close our connection to UbiOps


```python
api_client.close()
```

This tutorial just serves as an example. Feel free to reach out to our [support portal](https://www.support.ubiops.com) if you want to discuss your set-up in more detail.
