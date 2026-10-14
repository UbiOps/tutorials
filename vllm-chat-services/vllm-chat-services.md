# Deploy a vLLM server with Services

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/vllm-chat-services/vllm-chat-services/){ .md-button .md-button--primary } [View source code :fontawesome-brands-github:](https://github.com/UbiOps/tutorials/blob/master/vllm-chat-services/vllm-chat-services/vllm-chat-services.ipynb){ .md-button }

In this tutorial we will explain how to run a Llama LLM on UbiOps using [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) and [UbiOps Services](https://ubiops.com/docs/services/). The model used in this tutorial is the [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B) from Meta, which is freely available on HuggingFace and can be run with the vLLM engine on UbiOps. Hosting an LLM with UbiOps and exposing its endpoints through a server allows for a single LLM to process multiple requests concurrently, offering higher model throughput and accessibility for multiple users. 

With UbiOps services you can host a server-based application and expose it as a service in a simple manner. In this tutorial we expose the endpoints of our model through HTTP. This allows users to directly send HTTP requests to the model running in the deployment. The request method of the deployment is used to route request data to the `v1/chat/completions` endpoint, allowing for chatlike use cases.

### 1. Set up the UbiOps client

First, let's install the required packages and set up authentication.


```python
!pip install ubiops openai requests -qU
```

Now, we need to initialize all the necessary variables for the UbiOps deployment. To generate the API token you can follow [this guide](https://ubiops.com/docs/organizations/service-users/). 

A token for HuggingFace is needed, as well as permission to download the model that we will use for this tutorial. Follow these steps:
- Create a HuggingFace account. 
- Create a new user token, make sure *'Read access to contents of all public gated repos you can access'* is enabled. 
- Ask permission to use [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

Once you have an API token and a HuggingFace token with sufficient access, insert them below before continuing. Also fill in the name of your UbiOps project.

Whenever you change the instance type group, make sure it has access to a GPU, as this is required to run vLLM.


```python
## Add the name of your project and your API and HuggingFace token
API_TOKEN = "Token ..."  # Add your API token
HF_TOKEN = "hf_..."  # You need a token to download models from Huggingface
PROJECT_NAME = "..."  # Replace with your project name

## Set custom names if you want, please refrain from using underscores and spaces
ENVIRONMENT_NAME = "vllm-chat-v1"
DEPLOYMENT_NAME = "vllm-chat"
DEPLOYMENT_VERSION = "v1"
SERVICE_NAME = "vllm-chat"
PORT = 8888

## Change the instance type group if needed
INSTANCE_TYPE = "16384 MB + 4 vCPU + NVIDIA Ada Lovelace L4"  # You can find all possible Instance type groups in the WebApp under Project Admin > Project settings > Instance type groups
API_HOST_URL = "https://api.ubiops.com/v2.1"  # Standard UbiOps API URL is 'https://api.ubiops.com/v2.1', your URL may differ depending on your environment
```

Next, let's initialize the UbiOps client and check the connection.


```python
import ubiops

configuration = ubiops.Configuration(host=API_HOST_URL)
configuration.api_key['Authorization'] = API_TOKEN
client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)

status_check = api.service_status()

print(status_check)
print(f"Connected to UbiOps project '{PROJECT_NAME}'")
```

### 2. Prepare the environment and deployment package
In order to use vLLM inside the deployment, we need to set up the environment of the deployment so that everything will run smoothly. To do this we need to specify the `requirements.txt` and `ubiops.yaml`. More information on these files can be found in the [docs](https://ubiops.com/docs/environments/#uploading-dependency-information).

Before we start we create a directory to store our files in.


```python
import os

dir_name = "deployment_package"
os.makedirs(dir_name, exist_ok=True)
```

Python packages that you want to install with pip can be added to the `requirements.txt` file. 

Note that `vllm` automatically installs the CUDA drivers that are required to load the underlying model on a (NVIDIA) GPU.


```python
%%writefile {dir_name}/requirements.txt
vllm==0.13.0
openai
requests
```

Any OS packages that you might need can be added to the YAML file below.


```python
%%writefile {dir_name}/ubiops.yaml
apt:
  packages:
    - build-essential
    - python3-dev
    - libxcb1
```

Lastly, our deployment script with our vLLM engine. This script contains a `Deployment` class with two key methods:

- **`__init__` Method**  
  This method runs when the deployment starts. It fetches the model from HuggingFace, starts the vLLM as a subprocess, and opens an OpenAI-compatible server on port that we specify. We will connect to this port with a service to expose the endpoints of the vLLM on a server.

- **`request()` Method**  
  The request method contains the logic for processing incoming data. This method executes the calls that are being made to the UbiOps REST API endpoints, but now, because the endpoints of the vLLM will be exposed by Services, calls to the vLLM are handled instead. Currently, only a placeholder health check is inserted that checks the health of the server. When using Services, requests go directly to the vLLM server and bypass this method entirely.


**vLLM configuration**

The vLLM server is started with several flags. The `--max-model-len 128000` argument in vLLM, is chosen to fit within the maximum sequence length in tokens that the model can process for a single request. We set `--gpu-memory-utilization 0.9` to use 90% of available GPU memory. The `--host 0.0.0.0 --port 8888` flags expose the server on port 8888, which is required for UbiOps Services to connect.


```python
%%writefile {dir_name}/deployment.py
import os
import subprocess
import logging
import time
import requests
import torch
from openai import OpenAI

logging.basicConfig(level=logging.INFO)

class PublicError(Exception):
    def __init__(self, public_error_message):
        super().__init__()
        self.public_error_message = public_error_message

class Deployment:
    def __init__(self, base_directory, context):
        self.att_backend = os.getenv("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
        self.model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
        self.model_length = os.getenv("MAX_MODEL_LEN", 128000)
        self.vllm_gpu_memory_utilization = os.getenv("GPU_MEMORY_UTILIZATION", 0.9)
        self.port = int(os.getenv("VLLM_PORT", 8888))
        self.context = context

        if int(context["process_id"]) == 0:
            logging.info("Initializing vLLM server...")
            self.vllm_process = self.start_vllm_server()
            self.poll_health_endpoint()

        self.client = OpenAI(base_url=f"http://localhost:{self.port}/v1", api_key="...")

    def request(self, data, context):
        """
        Placeholder request method - returns server health status.
        When using Services, requests go directly to the vLLM server.
        """
        try:
            resp = requests.get(f'http://localhost:{self.port}/health', timeout=5)
            return {"status": "healthy", "status_code": resp.status_code}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def start_vllm_server(self):
        """
        Starts the vLLM server in a subprocess.
        """
        vllm_path = self.find_executable("vllm")
        # Build vLLM command
        vllm_cmd = [
            vllm_path, "serve",
            self.model_name,
            "--max-model-len", str(self.model_length),
            "--gpu-memory-utilization", str(self.vllm_gpu_memory_utilization),
            "--dtype", "float16",
            "--tensor-parallel-size", str(torch.cuda.device_count()),
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--attention-backend", str(self.att_backend),
        ]
        logging.info(f"Starting vLLM server: {' '.join(vllm_cmd)}")
        vllm_process = subprocess.Popen(vllm_cmd)
        logging.info("vLLM server starting...")
        return vllm_process

    def poll_health_endpoint(self):
        """
        Polls the /health endpoint to ensure the vLLM server is ready.
        """
        logging.info("Waiting for vLLM server to be ready...")
        while True:
            poll = self.vllm_process.poll()
            if poll is not None:
                logging.error("vLLM server process terminated unexpectedly.")
                raise RuntimeError(f"vLLM server exited with code: {poll}")
            try:
                resp = requests.get(f'http://localhost:{self.port}/health', timeout=5)
                if resp.status_code == 200:
                    logging.info("vLLM server is ready")
                    break
                else:
                    logging.warning(f"Unexpected status code: {resp.status_code}. Retrying...")
            except requests.exceptions.ConnectionError:
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
                time.sleep(5)

    def find_executable(self, executable_name):
        """
        Find the path to the vLLM executable.
        """
        result = subprocess.run(
            ['which', executable_name], 
            capture_output=True, 
            text=True, 
            check=False
        )
        path = result.stdout.strip()
        
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            logging.info(f"Found {executable_name} at: {path}")
            return path
        
        raise FileNotFoundError(f"{executable_name} not found in PATH")
```

Now we can create a .zip file that contains our Deployment script (the .py script), and the packages needed to create the environment (requirements.txt and the .yaml file). 


```python
import shutil

deployment_zip_path = shutil.make_archive(dir_name, 'zip', dir_name)
```

### 3 Create the deployment

Now we can create the [deployment] where we define the inputs and outputs of the model. We use input_type: "plain" and output_type: "plain" to accept and return JSON data. 

The input and output fields can be left empty. Traffic coming through the Service URL bypasses the standard `request()` method entirely and goes directly to your exposed port.


```python
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description="Deploying a chat model with vLLM",
    input_type="plain",
    output_type="plain",
    input_fields=[],
    output_fields=[],
)

deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)
```

### 4. Create the deployment version

A deployment version contains the actual code and resource configuration for your deployment. Each deployment can have multiple versions. For each version, you can deploy a different deployment script, environments, instance types, etc.

We configure:

- **Instance Type**: This was set at the start of this tutorial to `16384 MB + 4 vCPU + NVIDIA Ada Lovelace L4`, an instance type group with a GPU.
- **Scaling**: The minimum number of instances is set to 1. This means the instance will directly start running and continue to run if it is not turned off.
- **Request retention**: Full logging enabled for debugging and monitoring.
- **Idle time**: Maximum idle time is set to 900 seconds, which will keep the instance alive for 15 minutes after the last request.

**Note that when the instance is not turned off it will continuously consume resources and credits!**


```python
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    language='python3-12',
    instance_type_group_name=INSTANCE_TYPE,
    minimum_instances=1, # The deployment instance is continuously active when the minimum number of instances is set at 1
    maximum_instances=1,
    maximum_idle_time=900, # 15 minutes
    request_retention_mode="Full",
)

version = api.deployment_versions_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        data=version_template
)
```

To fetch the model from HuggingFace, the environment needs to be equipped with the HuggingFace token that we created. Here we create environment variables for the Huggingface token. We need this token to allow us to download models from gated HuggingFace repos.


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=ubiops.EnvironmentVariableCreate(name="HF_TOKEN", value=HF_TOKEN, secret=True),
)
```


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=ubiops.EnvironmentVariableCreate(name="VLLM_PORT", value=str(PORT), secret=False),
)
```

### 5. Upload the deployment package

Next we will upload our the .zip file of our deployment package that we just created and wait for the version to build. This can take a few minutes.

Logs of the build can be found in the Web App under Deployment > Logs or with `stream_logs=True` in the `wait_for_deployment_version` function. See `help(ubiops.utils.wait_for_deployment_version)` for help.


```python
## Upload deployment package
upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file=deployment_zip_path
)
print(upload_response)

## Check if the deployment is finished building. This can take a few minutes...
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=upload_response.revision,
    stream_logs=True,
)
```

### 6. Create the Service

Now that our deployment is ready, we can create a UbiOps Service to expose it with a public URL. Services provide automatic SSL, DNS, and authentication for your HTTP server. 

- The port that is exposed is `port=8888`. 
- For vLLMs, health monitoring can be achieved with the `/health` endpoint. 
- Authentication is provided by the UbiOps API token, which is activated with `authentication_required=True`. 
- Whether you want to log the requests that are being made to your service can be determined by setting the `request_storage` argument. 
- Limitations for the number of requests that can be made to your service per minute can be controlled with with the `rate_limiting_token` parameter.

More in depth explanations of this function can be found in the [documentation](https://ubiops.com/docs/services/).


```python
service_template = ubiops.ServiceCreate(
    name=SERVICE_NAME,
    deployment=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    port=PORT,  # Note that this port must match the port in the deployment.py
    health_check={"path":"/health"},
    request_storage_enabled=True,
    authentication_required=True,
    rate_limit_token=300,
)

service = api.services_create(
    project_name=PROJECT_NAME,
    data=service_template
)

print(f"Service '{SERVICE_NAME}' created successfully")
```

Using `services_get` method with the `endpoint` argument we can obtain the URL of our service. This URL can also be found in the Web App under Services.

Opening the `DOCS_URL` in your browser will provide you with the documentation of the API that we have now exposed.


```python
service = api.services_get(
    project_name=PROJECT_NAME,
    service_name=SERVICE_NAME
)

SERVICE_URL = service.endpoint
DOCS_URL = f"{service.endpoint}/docs"

print(f"API Docs: {DOCS_URL}")
```

### 7. Use the API with the Python requests library

Now let's get to work with the API of our model. We can do this by using the Python requests library, which will make requests to the server. First, we will do a health check to confirm that the server is running.


```python
import requests

headers = {"Authorization": API_TOKEN}

## Check if the server is online
try:
    response = requests.get(f"{SERVICE_URL}/health", headers=headers)
    print(f"Service status code: {response.status_code}")
    if response.status_code == 200:
        print("vLLM server is running and accessible!")
    else:
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"Connection test error: {e}")
```

Our service is now live and ready to accept requests. We can send and receive requests on the `/v1/chat/completions/` endpoint, allowing us to chat with our model. You can change the question below into something else. The header with our `API_TOKEN` will provide the authentication.


```python
import requests

headers = {"Authorization": API_TOKEN}

payload = {
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
        {
            "content": "You are a helpful assistant.",
            "role": "system"
        },
        {
            "content": "How many countries does the world count?",
            "role": "user"
        }
    ]
}

response = requests.post(
    f"{SERVICE_URL}/v1/chat/completions/",
    headers=headers,
    json=payload,
    timeout=60,
)

response.raise_for_status()  # Raises errors

data = response.json()
text = data["choices"][0]["message"]["content"]

print(text)
```

### 8. Use the API with OpenAI client

Let's now focus on OpenAI compatibility. Using the OpenAI client provides us with OpenAI-compatible endpoints that are the standard way to interact with LLM servers, which makes integrations with other platforms much easier.

First we need to set up an OpenAI client.


```python
from openai import OpenAI

client = OpenAI(
    api_key=API_TOKEN.lstrip("Token "),  # The Token prefix must be omitted for OpenAI compatibility
    base_url=f"{SERVICE_URL}/v1",
)
```

We need to specify the model before we can use it.


```python
openai_models = client.models.list()
model = openai_models.data[0].id

print(openai_models)
print(model)
```

A request can now be sent to our models API. You can change the contents if you want to ask a different question. Streaming is set to `False`, meaning the model's reponse will be singular instead of in chunks.


```python
stream_var = False

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Is OpenAI really non-profit?"}],
    stream=stream_var,
)

if stream_var:
    for chunk in response:
        if hasattr(chunk, 'choices') and chunk.choices:
            print(chunk.choices[0].delta.content, end="")
else:
    if hasattr(response, 'choices') and response.choices:
        print(response.choices[0].message.content)
```

### 9. Accessing the vLLM server endpoints from your browser

For accessing the vLLM server endpoints in your browser, you can use browser extension tools such as [Requestly](https://requestly.com/) to automatically inject authentication headers into requests. These tool will allow you to configure rules that add your Authorization header to all requests going to your service URL, enabling interaction with all available endpoints. You can find the vLLM API documentation with all the endpoints in your browser by navigating to the `/docs` endpoint at `https://{service_id}.services.ubiops.com/docs`.

[Requestly](https://requestly.io/) is a browser extension (available for Chrome, Firefox, Edge) that allows you to modify HTTP headers for specific URLs.

**Steps:**
1. Install the Requestly extension for your browser.
2. Choose HTTP Interceptor > Modify headers.
3. Configure the rule so that it includes your service URL ('services.ubiops.com').
4. Add a Request Header and choose 'authorization', fill the Header Value with the UbiOps token in the form 'Token ...'.
5. Save the rule.
6. You can now navigate to the endpoints in your browser.

## Summary

In this tutorial, you learned how to:

1. **Use UbiOps Services** to expose endpoints from a deployment on a server.
2. **Deploy an LLM model using vLLM** to host a model that provides OpenAI compatible chat completions.
3. **Test the API** using Python requests and the OpenAI client.
4. **Setup Requestly** to automatically authorize in your browser.

## Key Takeaways

- **UbiOps Services** lets you run any HTTP server (FastAPI, Flask, etc.) with automatic SSL, DNS, and load balancing
- The `request()` method becomes a placeholder when using services - all HTTP requests go directly to your server
- Services are ideal when you need custom endpoints, file uploads, or API compatibility (like OpenAI format)

## Next Steps

- Implement request logging and monitoring
- Scale to multiple instances for high availability
- Add support for batch processing of audio files
- Deploying other (multi-modal) models

## Resources

- [UbiOps Services Documentation](https://ubiops.com/docs/services/)
- [Llama Documentation](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [UbiOps Python Client](https://github.com/UbiOps/client-library-python)
- [OpenAI client](https://platform.openai.com/docs/libraries)
