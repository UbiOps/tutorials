# Deploy Whisper on UbiOps with vLLM and Services

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/whisper-vllm-tutorial/whisper-vllm-tutorial/){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/whisper-vllm-tutorial/whisper-vllm-tutorial/whisper-vllm-tutorial.ipynb){ .md-button .md-button--secondary }

In this tutorial, we will deploy OpenAI's Whisper model on UbiOps using vLLM's optimized serving framework. We'll expose the model through UbiOps Services, which allows direct HTTP access to the vLLM server's OpenAI-compatible API endpoints.

## What are UbiOps Services?

[UbiOps Services](https://ubiops.com/docs/services/) let you expose your deployments through custom HTTP endpoints. Unlike standard UbiOps deployment endpoints that follow the UbiOps API request/response structure, Services enable you to send direct HTTP requests to your deployments.

In this tutorial, we'll run a vLLM server using a [deployment package](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/) and expose it directly via a Service. This allows us to use the standard [OpenAI transcription API format](https://platform.openai.com/docs/guides/speech-to-text) without any UbiOps API wrapper. Services provide automatic HTTPS and TLS certificate provisioning, load balancing across deployment replicas, and integration with UbiOps monitoring, logging, and permissions.

## What is vLLM?

[vLLM](https://docs.vllm.ai/) is an easy-to-use, high-performance inference and serving framework for Large Language Models and audio models.

## Tutorial Overview

We will set up a connection with UbiOps, configure the deployment environment, create [deployment code](https://ubiops.com/docs/deployments/) that starts a vLLM server, deploy to UbiOps on a T4 GPU instance you can do so by going to Project Settings > Instance type (group) to see what instances you have enabled, create a Service to expose the vLLM API, and test transcription.

For demo purposes, we will deploy a vLLM server that hosts the [openai/whisper-small](https://huggingface.co/openai/whisper-small) model. To follow along, ensure that your UbiOps subscription contains GPUs.

## 1. Set up a connection with the UbiOps API client

First, we'll install the [UbiOps Python Client Library](https://ubiops.com/docs/python_client_library/) and initialize our connection to UbiOps.


```python
!pip install -qU ubiops openai requests
```

Now, we will need to initialize all the necessary variables for the UbiOps deployment and the deployment directory, which we will zip and upload to UbiOps.


```python
# Initialize variables
API_TOKEN = "<INSERT API TOKEN WITH PROJECT EDITOR RIGHTS>"
PROJECT_NAME = "<INSERT YOUR PROJECT NAME>"
API_HOST_URL = "<INSERT YOUR HOST API URL>" # Standard UbiOps API URL is 'https://api.ubiops.com/v2.1', your URL may differ depending on your environment

DEPLOYMENT_NAME = "whisper-vllm"
DEPLOYMENT_VERSION = "v1"
SERVICE_NAME = "whisper-service"

print(f"Your deployment will be named: {DEPLOYMENT_NAME}")
print(f"Your service will be named: {SERVICE_NAME}")
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


```python
# Create deployment package directory
import os

dir_name = "deployment_package"
os.makedirs(dir_name, exist_ok=True)
print(f"Created directory: {dir_name}")
```

## 2. Setup deployment environment

We'll configure the deployment environment with the necessary dependencies and system packages. This is done through two files: [requirements.txt](https://ubiops.com/docs/howto/howto-requirements-txt/) for Python packages and [ubiops.yaml](https://ubiops.com/docs/environments/ubiops-yaml/) for system-level configuration.

### requirements.txt

The requirements.txt file specifies Python packages to install. We use `vllm[audio]` which includes vLLM with audio processing dependencies like librosa and soundfile. The `openai` package is included for testing the OpenAI-compatible API, and `requests` is used for HTTP requests and health checks.


```python
%%writefile {dir_name}/requirements.txt
vllm[audio]
openai
requests
```

### ubiops.yaml

The [ubiops.yaml](https://ubiops.com/docs/environments/ubiops-yaml/) file configures system-level dependencies and environment variables. We install `build-essential` and `python3-dev` for C/C++ compilation tools needed by some Python packages, and `ffmpeg` for audio processing which Whisper uses for audio resampling.


```python
%%writefile {dir_name}/ubiops.yaml
apt:
  packages:
    - build-essential
    - python3-dev
    - ffmpeg
```

## 3. Creating UbiOps deployment code

### Understanding the Deployment Class

The [deployment code](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/) consists of a `Deployment` class with two key methods. The `__init__` method runs once when the deployment starts and is used to set up environment variables for vLLM, start the vLLM server as a subprocess, and wait for the server to be healthy before accepting requests. The `request()` method acts as a placeholder that returns server health status if called directly. When using Services, requests go directly to the vLLM server and bypass this method entirely.

### vLLM Server Configuration

The vLLM server is started with several flags. The `--task transcription` flag configures vLLM specifically for audio transcription, which is required for Whisper models as explained in the [vLLM transcription documentation](https://docs.vllm.ai/en/latest/contributing/model/transcription/). We use `--dtype float16` for FP16 precision which is neccessary for this gpu due to the older architecture that it's built on. The `--max-model-len 448` argument in vLLM, when used with transcription models like Whisper, sets the maximum sequence length (in tokens) that the model can process for a single request. We set `--gpu-memory-utilization 0.9` to use 90% of available GPU memory. The `--host 0.0.0.0 --port 8080` flags expose the server on port 8080, which is required for UbiOps Services to connect.


```python
%%writefile {dir_name}/deployment.py
import os
import subprocess
import logging
import time
import requests
import torch

logging.basicConfig(level=logging.INFO)


class Deployment:
    def __init__(self, base_directory, context):
        os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
        
        # Model configuration
        self.model_name = "openai/whisper-large-v3-turbo"
        self.task = os.getenv("WHISPER_TASK", "transcription")  # or "translation", if the model supports it
        self.max_model_len = int(os.getenv("MAX_MODEL_LEN", "448"))

        # Start vLLM server
        logging.info("Initializing vLLM server for Whisper...")
        self.vllm_process = self.start_vllm_server()
        self.wait_for_server()
        logging.info("vLLM Whisper server is ready!")

    def request(self, data):
        """
        Placeholder request method - returns server health status.
        When using Services, requests go directly to the vLLM server.
        """
        try:
            resp = requests.get('http://localhost:8080/health', timeout=5)
            return {"status": "healthy", "status_code": resp.status_code}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def start_vllm_server(self):
        """
        Starts the vLLM server for Whisper in a subprocess.
        """
        vllm_path = self.find_executable("vllm")
        
        # Build vLLM command
        vllm_cmd = [
            vllm_path, "serve",
            self.model_name,
            "--task", self.task,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", "0.9",
            "--dtype", "float16",
            "--tensor-parallel-size", str(torch.cuda.device_count()),
            "--host", "0.0.0.0",
            "--port", "8080"  # Service will connect to this port, you can modify it accordingly 
        ]
        
        logging.info(f"Starting vLLM server: {' '.join(vllm_cmd)}")
        vllm_process = subprocess.Popen(vllm_cmd)
        logging.info("vLLM server starting...")
        
        return vllm_process

    def wait_for_server(self):
        """
        Wait until the vLLM server is ready to accept requests.
        """
        logging.info("Waiting for vLLM server to be ready...")
        max_retries = 60
        
        for retry_count in range(max_retries):
            # Check if process crashed
            poll = self.vllm_process.poll()
            if poll is not None:
                logging.error("vLLM server process terminated unexpectedly.")
                raise RuntimeError(f"vLLM server exited with code: {poll}")
            
            # Try health check
            try:
                resp = requests.get('http://localhost:8080/health', timeout=5)
                if resp.status_code == 200:
                    logging.info("vLLM server is ready!")
                    return
            except requests.exceptions.RequestException:
                time.sleep(5)
        
        raise RuntimeError("vLLM server failed to start within timeout period")

    @staticmethod
    def find_executable(executable_name):
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

## 4. Create UbiOps deployment

### Deployment Configuration

We create a [deployment](https://ubiops.com/docs/deployments/) with `input_type: "plain"` and `output_type: "plain"` to accept and return JSON data. However, when accessed through Services, these input/output types don't matter because requests bypass the UbiOps API structure and go directly to the vLLM server.

If you try the `request()` method it will return the server health status if called directly as we configured it earlier in the `deployment.py` file. When using Services, requests go directly to the vLLM server and bypass this method entirely.


```python
# Create deployment
deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data={
        "name": DEPLOYMENT_NAME,
        "description": "Whisper small with vLLM",
        "input_type": "plain",
        "output_type": "plain",
    }
)
print(f"Created deployment: {deployment.name}")
```

### Deployment Version Configuration

We create a [deployment version](https://ubiops.com/docs/deployments/deployment-versions/) with specific settings. We use Python 3.12 as the runtime environment and, for example, an NVIDIA T4 (Tesla Architecture). We set `maximum_instances: 1` and `minimum_instances: 0` to allow the deployment to scale to zero when idle, saving costs. The `maximum_idle_time: 900` keeps the instance alive for 15 minutes after the last request. We add labels to mark the deployment as OpenAI-compatible for easier discovery.

**Note** : Make sure you have that instance type by going to Project Settings > Instance type (group) pages to check what compute you have available.


```python
# Create deployment version
version_template = {
    "version": DEPLOYMENT_VERSION,
    "environment": "python3-12",
    "instance_type_group_name": "12288 MB + NVIDIA Tesla T4",  # T4 GPU instance
    "maximum_instances": 1,
    "minimum_instances": 0,
    "maximum_idle_time": 900,  # 15 minutes
    "labels": {
        "openai-compatible": "true",
        "openai-model-names": "openai/whisper-small",
        "model-type": "speech-to-text"
    }
}

deployment_version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=version_template,
)

print(f"Created deployment version: {deployment_version.version}")
```

### Adding custom environment variables

You can configure the deployment by adding environment variables such as `WHISPER_TASK` for "transcription" or "translation", or `MAX_MODEL_LEN` for audio sequence length (default: 448) which lets you modify the deployment without having to upload a revision. Note that no [HuggingFace](https://huggingface.co/) token is needed as Whisper models are public and ungated but one can be set up as an environment variable if you want to use a model that is available there.

The critical environment variable here is `VLLM_ATTENTION_BACKEND=TRITON_ATTN`. This forces vLLM to use the [Triton attention backend](https://docs.vllm.ai/en/v0.10.1/api/vllm/v1/attention/backends/triton_attn.html), which is essential for Whisper's encoder-decoder cross-attention pattern. Without this setting, vLLM will fail to start on T4 GPUs with Whisper models due to incompatibility between the default attention backend and the encoder-decoder architecture.


```python

env_vars = [
    {"name": "VLLM_ATTENTION_BACKEND", "value": "TRITON_ATTN"},
    {"name": "WHISPER_TASK", "value": "transcription"},
    {"name": "MAX_MODEL_LEN", "value": "448"}
]

for env_var in env_vars:
    env_data = ubiops.EnvironmentVariableCreate(
        name=env_var["name"],
        value=env_var["value"],
        secret=False
    )
    
    api.deployment_version_environment_variables_create(
        PROJECT_NAME,
        DEPLOYMENT_NAME,
        DEPLOYMENT_VERSION,
        env_data
    )
    print(f"Created environment variable: {env_var['name']}")
```

## 5. Archive and upload deployment

Now we package and upload our deployment code to UbiOps. This will trigger a build process that installs all dependencies, and prepares the deployment for execution. Building can take 10-15 minutes due to the size of vLLM and the other dependencies.

**Note:** To check the progress you can either check the UI, or uncomment `stream_logs=True` in the `wait_for` method to see the logs in the notebook.


```python
import shutil

# Archive the deployment directory
deployment_zip_path = shutil.make_archive(dir_name, 'zip', dir_name)
print(f"Created archive: {deployment_zip_path}")
```


```python
# Upload deployment package
upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file=deployment_zip_path
)
print(f"Upload started. Revision ID: {upload_response.revision}")
```


```python
# Wait for deployment to be ready
print("Waiting for deployment build to complete...")
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=upload_response.revision,
    # stream_logs = True
)
print("Deployment is ready!")
```

## 6. Create a Service to expose the vLLM API

### Understanding Services

Now we'll create a [UbiOps Service](https://ubiops.com/docs/services/) that exposes our vLLM server to the internet. The Service connects to port 8080 where vLLM is running, provides a public HTTPS endpoint, automatically handles TLS certificates, and load balances across deployment replicas.

### Service Configuration

The service requires a name, the deployment to connect to, the specific version to use, and the port number (8080) which must match the port vLLM listens on. After creation, the service will be accessible at `https://[service-id].services.ubiops.com`. This URL will route directly to your vLLM server's OpenAI-compatible API. We also specify the `health_check_path` to point to the `/health` endpoint so UbiOps can monitor service availability. For authentication, we configure the service to require a UbiOps API token passed in the `Authorization` header of requests. This ensures only authorized users can access the service, leveraging UbiOps' existing permission system to control who can make requests.


```python
# Create a Service
service = api.services_create(
    project_name=PROJECT_NAME,
    data={
        "name": SERVICE_NAME,
        "deployment": DEPLOYMENT_NAME,
        "version": DEPLOYMENT_VERSION,
        "port": 8080,  # Port where vLLM server listens
        "health_check_path" : "/health",
        "health_cheack_interval" : 30,
        "request_storage_enabled" : True,
        "authentication_required" : True
    }
)

SERVICE_URL = f"https://{service.id}.services.ubiops.com"

print(f"Service created: {service.name}")
print(f"Service ID: {service.id}")
print(f"\n Service URL: {SERVICE_URL}")
print(f"\n Transcription endpoint: {SERVICE_URL}/v1/audio/transcriptions")
```

## 7. Test the transcription service

Now let's test our Whisper service by sending audio files to the transcription endpoint and getting back the transcribed text.

### Test 1: Check server health


```python
import requests

headers = {
    "Authorization": f"{API_TOKEN}"
}

# Test health endpoint
health_response = requests.get(f"{SERVICE_URL}/health", headers=headers)
print(f"Server health: {health_response.status_code}")
print(health_response.text)
```

### Test 2: Check available models


```python
# Check available models
models_response = requests.get(f"{SERVICE_URL}/v1/models", headers=headers)
print(f"Available models: {models_response.json()}")
```

### Test 3: Transcribe an audio file

For this test, you'll need an audio file in formats like .wav, .mp3, or .m4a. Replace `"your_audio.wav"` with the path to your audio file. If you don't have a file you can find one here: https://www.kaggle.com/datasets/pavanelisetty/sample-audio-files-for-speech-recognition.


```python
# Transcribe audio file
audio_file_path = "your_audio.wav"  # Replace with your audio file path

with open(audio_file_path, "rb") as audio_file:
    response = requests.post(
        f"{SERVICE_URL}/v1/audio/transcriptions",
        files={"file": ("audio.wav", audio_file, "audio/wav")},
        headers=headers,
        data={
            "model": "openai/whisper-small",
            "language": "en",
        }
    )

print("Transcription result:")
print(response.json())
```

### Test 4: cURL command for transcription


```python
# cURL transcription test
!curl -X POST \
  -H "Authorization: {API_TOKEN}" \
  -F "file=@your_audio.wav" \
  -F "model=openai/whisper-small" \
  -F "language=en" \
  -F "response_format=json" \
  {SERVICE_URL}/v1/audio/transcriptions
```

### Test 5: Using OpenAI Python client

The service is OpenAI-compatible, so you can use the [official OpenAI Python client](https://platform.openai.com/docs/guides/speech-to-text).


```python
from openai import OpenAI

# Initialize OpenAI client pointing to your service
client = OpenAI(
    api_key="dummy",  # Not needed for UbiOps Services
    base_url=f"{SERVICE_URL}/v1"
)

# Transcribe audio
with open("your_audio.wav", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="openai/whisper-small",
        headers=headers,
        file=audio_file,
        language="en"
    )

print(f"Transcription: {transcription.text}")
```

### Test 7: Multilingual transcription

Here we can try and translate an audio in a different language by setting the `language` parameter. Whisper [supports 99 languages](https://github.com/openai/whisper#available-models-and-languages), the parameter is set to `es` for spanish but can be changed depending on your audio file and languages supported.


```python
# Auto-detect language
with open("your_audio.wav", "rb") as audio_file:
    response = requests.post(
        f"{SERVICE_URL}/v1/audio/transcriptions",
        files={"file": audio_file},
        headers=headers,
        data={
            "model": "openai/whisper-small",
            # No language specified - will auto-detect
        }
    )
    result = response.json()
    print(f"Detected language: {result.get('language')}")
    print(f"Text: {result.get('text')}")

# Transcribe Spanish audio
# with open("spanish_audio.wav", "rb") as audio_file:
#     response = requests.post(
#         f"{SERVICE_URL}/v1/audio/transcriptions",
#         files={"file": audio_file},
#         data={
#             "model": "openai/whisper-small",
#             "language": "es"  # Spanish
#         }
#     )
#     print(response.json())
```

### Accessing API Documentation with Browser Authentication

The vLLM API documentation is available at `/docs` endpoint. However, since authentication is required at the UbiOps Service level, you'll need to inject the Authorization header using a browser extension.

### Using Requestly Browser Extension

[Requestly](https://requestly.io/) is a browser extension (available for Chrome, Firefox, Edge) that allows you to modify HTTP headers for specific URLs.

**Steps:**
1. Install the Requestly extension for your browser
2. Choose HTTP Interceptor > Modify headers
3. Configure the rule so that it includes your service URL ('services.ubiops.com')
4. Add a Request Header and choose 'authorization', fill the Header Value with the UbiOps token in the form 'Token ...'
5. Save the rule
6. You can now navigate to the endpoints in your browser 

**Example configuration:**
```
URL Pattern: your-service-id.services.ubiops.com
Header Name: Authorization
Header Value: Token your-api-token-here
```

Once configured, you can access:
- ReDoc: `{SERVICE_URL}/docs`


**Note:** The exact workflow for creating rules differs per browser and Requestly version. Refer to [Requestly's documentation](https://docs.requestly.com/general/getting-started/introduction) for browser-specific instructions.

## 9. Cleanup

When you're done testing, scale down the deployment version to avoid extra charges:


```python

raise SystemExit("Stopped from running all cells to avoid scaling down the deployment before completing all chapters.\nYou can execute the next cells manually.")
```


```python
version_template = ubiops.DeploymentVersionUpdate(
    minimum_instances=0
)

deployment_version = api.deployment_versions_update(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=version_template
)
```

Now we can close the api client.


```python
# Close API client
client.close()
print("Closed UbiOps connection")
```
