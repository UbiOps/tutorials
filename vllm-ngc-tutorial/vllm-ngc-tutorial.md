# Deploy an Nvidia optimized vLLM Server on UbiOps with Docker and Services

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/vllm-ngc-tutorial/vllm-ngc-tutorial/){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/vllm-ngc-tutorial/vllm-ngc-tutorial/vllm-ngc-tutorial.ipynb){ .md-button .md-button--secondary }

In this tutorial, we will deploy NVIDIA's vLLM inference server on UbiOps using a custom Docker image. We'll expose the server through UbiOps Services, which allows direct HTTP access to the vLLM OpenAI-compatible API endpoints. For this example we will be deploying the model [nvidia/Llama-3.1-8B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8) which is the quantized version of the Meta's Llama 3.1 8B Instruct model.

## What are UbiOps Services?

[UbiOps Services](https://ubiops.com/docs/services/) let you expose your deployments through custom HTTP endpoints. Unlike standard UbiOps deployment endpoints that follow the UbiOps API request/response structure, Services enable you to send direct HTTP requests to your deployments. This is particularly useful when combined with our [bring your own Docker image](https://ubiops.com/docs/deployments/docker-support/) feature, as it allows you to deploy a range of server-based applications such as vLLM, Ollama, or custom servers built with Flask, FastAPI, and others and expose it directly via a service.

In this tutorial, we'll use the optimized NVIDIA NGC vLLM Docker image and expose it directly via a Service. Services provide automatic HTTPS and TLS certificate provisioning, load balancing across deployment replicas, and integration with UbiOps monitoring, logging, and permissions.

## What is vLLM?

[vLLM](https://github.com/vllm-project/vllm) is a fast and easy-to-use library for large language model (LLM) inference and serving. It provides high-throughput serving through things like PagedAttention, continuous batching, and optimized CUDA kernels. vLLM supports a wide range of Hugging Face models and provides an OpenAI-compatible API server.

## What is the NVIDIA NGC vLLM Container?

The [NVIDIA NGC vLLM container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm) is an optimized container image that includes vLLM with NVIDIA-specific optimizations for better performance on NVIDIA GPUs. The container comes pre-configured with all necessary dependencies, CUDA libraries, and optimizations for multiple NVIDIA GPU architectures.

## Tutorial Overview

We will set up a connection with UbiOps, create a custom [environment](https://ubiops.com/docs/environments/) with the NVIDIA NGC vLLM Docker image, create a [deployment](https://ubiops.com/docs/deployments/) that runs the vLLM server, create a Service to expose the OpenAI-compatible API, and test LLM inference with various models and configurations.

For this tutorial, we'll use the official [NVIDIA NGC vLLM container (version 25.10-py3)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm/tags?version=25.10-py3). To follow along, ensure you have [Docker Engine](https://docs.docker.com/engine/) or [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed locally to pull and save the image, and that your UbiOps subscription has access to custom environments and GPU instances.

## 1. Set up a connection with the UbiOps API client

First, we'll install the [UbiOps Python Client Library](https://ubiops.com/docs/python_client_library/) and initialize our connection to UbiOps.


```python
!pip install -qU ubiops requests openai
```

Now, we will need to initialize all the necessary variables for the UbiOps deployment. 

See [here](https://ubiops.com/docs/organizations/service-users/) to learn how you can get these variables.


```python
# Initialize variables
API_TOKEN = "<INSERT API TOKEN WITH PROJECT EDITOR RIGHTS>"
PROJECT_NAME = "<INSERT YOUR PROJECT NAME>"
API_HOST_URL = "<INSERT YOUR HOST API URL>"  # The UbiOps SaaS API URL is 'https://api.ubiops.com/v2.1'

DEPLOYMENT_NAME = "vllm-ngc-server"
DEPLOYMENT_VERSION = "v1"
SERVICE_NAME = "vllm-ngc-service"
ENVIRONMENT_NAME = "vllm-ngc-environment"

MODEL_NAME = "nvidia/Llama-3.1-8B-Instruct-FP8"

print(f"Your deployment will be named: {DEPLOYMENT_NAME}")
print(f"Your service will be named: {SERVICE_NAME}")
print(f"Your environment will be named: {ENVIRONMENT_NAME}")
print(f"Model to deploy: {MODEL_NAME}")
```


```python
# Initialize UbiOps client
import ubiops

configuration = ubiops.Configuration(host=f"{API_HOST_URL}")
configuration.api_key["Authorization"] = API_TOKEN

client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)

# Test connection
api.service_status()
```

## 2. Building the vLLM Docker image

We'll build a custom Docker image based on the [NVIDIA NGC vLLM container (version 25.10-py3)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm/tags?version=25.10-py3). This container is pre-optimized with NVIDIA-specific performance enhancements and includes all necessary dependencies for running vLLM on NVIDIA GPUs.

### Create Dockerfile

First, let's create a Dockerfile that configures the vLLM server with the appropriate settings. 


```python
%%writefile Dockerfile
FROM nvcr.io/nvidia/vllm:25.10-py3

# Run vLLM OpenAI API server with optimized settings
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", \
            "--model", "nvidia/Llama-3.1-8B-Instruct-FP8", \
            "--trust-remote-code", \
            "--tensor-parallel-size", "1", \
            "--max-model-len", "4096", \
            "--gpu-memory-utilization", "0.90", \
            "--host", "0.0.0.0", \
            "--port", "8000"]
```

### Pull base image and build

Now we'll pull the NVIDIA NGC vLLM base image and build our custom image. The NGC container is optimized for NVIDIA GPUs and includes the latest performance improvements.


```python
# Pull the NVIDIA NGC vLLM image
!docker pull nvcr.io/nvidia/vllm:25.10-py3
```


```python
# Build the Docker image
!docker build -t vllm-ngc-server:latest .
```

### Save the Docker image

After building, we need to save the Docker image as a compressed tar archive. This archive will be uploaded to UbiOps as a [custom environment](https://ubiops.com/docs/environments/#bring-your-own-docker-image). The save process exports the entire image including all layers, which may take a few minutes depending on the image size.


```python
# Save the Docker image as a tar archive
!docker save vllm-server:latest -o vllm-ngc-server.tar
```

Now we can zip the image.


```python
!gzip vllm-ngc-server.tar
```

## 3. Creating a custom environment in UbiOps

Now we'll create a [custom environment](https://ubiops.com/docs/environments/#bring-your-own-docker-image) in UbiOps and upload the vLLM Docker image to it. Custom environments allow you to bring your own Docker images with pre-configured software stacks, libraries, and runtime configurations. This is ideal for deploying containerized applications like vLLM that have specific dependencies and server configurations already built into the image.

### Create the environment

First, we create an environment resource in UbiOps. The environment acts as a container for our Docker image and can be reused across multiple deployments.


```python
# Create custom environment
environment = api.environments_create(
    project_name=PROJECT_NAME,
    data=ubiops.EnvironmentCreate(
        name=ENVIRONMENT_NAME,
        display_name="NVIDIA NGC vLLM Environment",
        description="Custom environment with NVIDIA NGC vLLM 25.10 container for LLM inference",
        supports_request_format=False  # Docker image handles its own request format
    )
)

print(f"Created environment: {environment.name}")
print(f"Environment ID: {environment.id}")
```

### Upload the Docker image

Now we upload the Docker image tar archive to the environment. This process may take several minutes depending on the image size and your internet connection speed. UbiOps will extract and prepare the image for use in deployments.


```python
# Upload the Docker image archive
with open("vllm-ngc-server.tar.gz", "rb") as f:
    upload_response = api.environment_revisions_file_upload(
        project_name=PROJECT_NAME,
        environment_name=ENVIRONMENT_NAME,
        file=f
    )

print(f"Uploaded Docker image to environment")
print(f"Revision ID: {upload_response.revision}")
```

### Wait for environment build to complete

The environment needs to be built before we can use it in a deployment. This process extracts and prepares the Docker image layers.


```python
ubiops.utils.wait_for_environment(
    api.api_client,
    PROJECT_NAME,
    ENVIRONMENT_NAME,
    timeout=1800,
    quiet=False,
    stream_logs=False
)
```

## 4. Create a deployment

Now we'll create a [deployment](https://ubiops.com/docs/deployments/) that will run our vLLM server. Deployments in UbiOps are scalable computational units that can process requests. When using custom Docker images with Services, the deployment acts as a container orchestrator that manages your server instances.


```python
# Create deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description="vLLM inference server with NVIDIA NGC container",
    input_type="plain",
    output_type="plain",
    supports_request_format=False,  # We'll use the server's native API format
    labels={"type": "vllm-server", "provider": "nvidia-ngc"}
)

deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)

print(f"Created deployment: {deployment.name}")
```

## 5. Create a deployment version

We create a deployment version using our custom environment. The key difference from a standard deployment is that we specify our custom environment name instead of a standard Python environment. We set minimum_instances: 1 to keep at least one instance running because cold-starts can take a while and we want a quick response for this tutorial.



```python

# Create deployment version with custom environment
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment=ENVIRONMENT_NAME,  # Use our custom environment
    instance_type="16384 MB + 4 vCPU + NVIDIA Ada Lovelace L4",  
    minimum_instances=1,  # Keep at least one instance warm
    request_retention_mode="full"  # Store request logs + I/O
)

deployment_version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=version_template
)

print(f"Created deployment version: {deployment_version.version}")
print(f"Using environment: {deployment_version.environment}")
```

## 6. Create a Service

Now we'll create a [Service](https://ubiops.com/docs/services/) that exposes our vLLM deployment through a custom HTTP endpoint. Services provide a direct way to access your deployment's API without going through the standard UbiOps request format. This is perfect for vLLM's OpenAI-compatible API server.

The Service will:
- Provide an HTTPS endpoint with automatic TLS certificate provisioning
- Load balance requests across deployment instances
- Integrate with UbiOps authentication and monitoring
- Forward requests directly to vLLM's API server


```python
# Create service
service_template = ubiops.ServiceCreate(
    name=SERVICE_NAME,
    description="Service exposing vLLM OpenAI-compatible API",
    service_type="deployment",
    deployment=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    port=8000
)

service = api.services_create(
    project_name=PROJECT_NAME,
    data=service_template
)

SERVICE_URL = api.services_get(project_name=PROJECT_NAME, service_name=SERVICE_NAME)

print(f"Service created: {service.name}")
print(f"Service ID: {service.id}")
print(f"Service URL: {SERVICE_URL}")
```

## 7. Wait for deployment to be ready

Before we can test the vLLM server, we need to wait for the deployment to start and the model to be loaded. The first startup takes longer because the deployment instance needs to spin up and the model needs to be loaded into GPU memory.


This can take anywhere from 5-30 minutes depending on model size and network speed.


```python

ubiops.utils.wait_for_deployment_version(
    api.api_client,
    PROJECT_NAME,
    DEPLOYMENT_NAME,
    DEPLOYMENT_VERSION,
    timeout=3600,
    quiet=False,
    stream_logs=True
)
```

## 8. Test the vLLM API

Now that our vLLM server is running, we can test it using the OpenAI Python client. The vLLM server provides an OpenAI-compatible API, so we can use the OpenAI SDK to interact with it.

### Test 1: Health check


```python
import requests

# Test health endpoint
response = requests.get(
    f"{SERVICE_URL}/health",
    headers={"Authorization": API_TOKEN}
)

print(f"Response: {response}")
```

### Test 2: List available models


```python
# List available models
response = requests.get(
    f"{SERVICE_URL}/v1/models",
    headers={"Authorization": API_TOKEN}
)

if response.status_code == 200:
    models = response.json()
    print("Available models:")
    for model in models['data']:
        print(f"  - {model['id']}")
```

### Test 3: Chat completion

Let's test the chat completion endpoint with a simple question.


```python
from openai import OpenAI

# Initialize OpenAI client pointing to our vLLM service
client = OpenAI(
    base_url=f"{SERVICE_URL}/v1",
    api_key=API_TOKEN
)

print("Test: Chat completion")
print("=" * 60)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(f"Response: {response.choices[0].message.content}")
print(f"\nToken usage:")
print(f"  Prompt tokens: {response.usage.prompt_tokens}")
print(f"  Completion tokens: {response.usage.completion_tokens}")
print(f"  Total tokens: {response.usage.total_tokens}")
```

### Test 4: Streaming response

vLLM supports streaming responses, which allows you to receive tokens as they're generated rather than waiting for the entire response.


```python
print("\nTest: Streaming chat completion")
print("=" * 60)

stream = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": "Write a haiku about machine learning."}
    ],
    max_tokens=100,
    stream=True
)

print("Streaming response: ", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")
```

### Test 5: Different temperature settings

Test how temperature affects response diversity.


```python
print("Test: Temperature comparison")
print("=" * 60)

prompt = "Complete this sentence: The future of the world is"

for temp in [0.0, 0.5, 1.0]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=temp
    )
    print(f"\nTemperature {temp}:")
    print(f"{response.choices[0].message.content}")
```

### Test 6: Using cURL

You can also test the API using cURL commands directly.


```python
# Test with cURL
import json

curl_command = f'''curl {SERVICE_URL}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: {API_TOKEN}" \\
  -d '{json.dumps({
    "model": MODEL_NAME,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hello!"}
    ],
    "max_tokens": 50
  })}'
'''

print("cURL command to test the API:")
print(curl_command)
```


```python
# Execute the cURL command via requests
response = requests.post(
    f"{SERVICE_URL}/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": API_TOKEN
    },
    json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"}
        ],
        "max_tokens": 50
    }
)

print("\nAPI response:")
print(json.dumps(response.json(), indent=2))
```

## 9. Accessing API documentation

The vLLM server provides OpenAI-compatible API documentation. However, since authentication is required at the UbiOps Service level, you'll need to inject the Authorization header using a browser extension to view the interactive documentation.

### Using Requestly Browser Extension

[Requestly](https://requestly.io/) is a browser extension (available for Chrome, Firefox, Edge) that allows you to modify HTTP headers for specific URLs.

**Steps:**
1. Install the Requestly extension for your browser
2. Choose HTTP Interceptor > Modify headers
3. Configure the rule so that it includes your service URL ('services.ubiops.com')
4. Add a Request Header and choose 'Authorization', fill the Header Value with the UbiOps token in the form 'Token ...'
5. Save the rule
6. You can now navigate to the API documentation endpoints in your browser

**Example configuration:**
```
URL Pattern: *.services.ubiops.com
Header Name: Authorization
Header Value: Token your-api-token-here
```

Once configured, you can access:
- OpenAPI docs: `{SERVICE_URL}/docs`
- API specification: `{SERVICE_URL}/openapi.json`

**Note:** The exact workflow for creating rules differs per browser and Requestly version. Refer to [Requestly's documentation](https://docs.requestly.com/general/getting-started/introduction) for browser-specific instructions.


```python
print(f"API Documentation URLs:")
print(f"  OpenAPI docs: {SERVICE_URL}/docs")
print(f"  OpenAPI spec: {SERVICE_URL}/openapi.json")
print(f"\nRemember to configure Requestly to add the Authorization header!")
```

## 10. Cleanup

When you're done testing, clean up your resources to avoid extra charges. The statement below was added to prevent scaling down the instance when running the full notebook at once.


```python
raise SystemExit("Prevented running all cells to avoid scaling down the deployment before completing all chapters. Please execute the last cell manually to scale down the instance.")
```


```python
# Scale down to 0 instances
version_template = ubiops.DeploymentVersionUpdate(
    minimum_instances=0
)

deployment_version = api.deployment_versions_update(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=version_template
)

print("Scaled deployment to 0 instances")
```

Now we can close the API client.


```python
# Close API client
client.close()
print("Closed UbiOps connection")
```


We have successfully deployed a vLLM inference server on UbiOps using the NVIDIA NGC optimized container. This tutorial demonstrates how to deploy server-based LLM inference applications on UbiOps using bring-your-own-Docker and Services.

Feel free to reach out to our [support portal](https://www.support.ubiops.com) if you want to discuss your set-up in more detail or need help with advanced configurations.
