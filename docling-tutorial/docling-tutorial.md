# Deploy Docling Serve on UbiOps with Docker and Services

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/docling-tutorial/docling-tutorial/){ .md-button .md-button--primary } [View source code :fontawesome-brands-github:](https://github.com/UbiOps/tutorials/blob/master/docling-tutorial/docling-tutorial/docling-tutorial.ipynb){ .md-button }.

In this tutorial, we will deploy IBM's Docling Serve document processing server on UbiOps using a custom Docker image. We'll expose the server through UbiOps Services, which allows direct HTTP access to the Docling API endpoints.

## What are UbiOps Services?

[UbiOps Services](https://ubiops.com/docs/services/) let you expose your deployments through custom HTTP endpoints. Unlike standard UbiOps deployment endpoints that follow the UbiOps API request/response structure, Services enable you to send direct HTTP requests to your deployments. This is particularly useful when combined with our [bring your own Docker image](https://ubiops.com/docs/deployments/docker-support/) feature, as it allows you to deploy a range of server-based application such as Docling Serve, Ollama, or custom servers built with Flask, FastAPI, and others and expose it directly via a service.

In this tutorial, we'll use a pre-built Docker image from the [Docling Serve project](https://github.com/docling-project/docling-serve) and expose it directly via a Service. Services provide automatic HTTPS and TLS certificate provisioning, load balancing across deployment replicas, and integration with UbiOps monitoring, logging, and permissions.

## What is Docling?

[Docling](https://github.com/DS4SD/docling) is an advanced document processing toolkit developed by IBM Research that can parse and convert various document formats including PDF, DOCX, PPTX, images, HTML, and more into structured formats like Markdown and JSON. It provides high-quality document understanding capabilities including layout analysis, table extraction, optical character recognition (OCR), and document structure detection.

## What is Docling Serve?

[Docling Serve](https://github.com/docling-project/docling-serve) is a FastAPI-based REST API server that wraps Docling's document processing capabilities. It provides endpoints for converting documents, checking server health, and managing processing jobs. The server is containerized and ready to deploy with Docker.

## Tutorial Overview

We will set up a connection with UbiOps, create a custom [environment](https://ubiops.com/docs/environments/) with the Docling Serve Docker image, create a [deployment](https://ubiops.com/docs/deployments/) that runs the Docling server, create a Service to expose the Docling API, and test document processing with various file formats.

For this tutorial, we'll use the official [Docling Serve Docker image](https://github.com/docling-project/docling-serve). To follow along, ensure you have [Docker Engine](https://docs.docker.com/engine/) or [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed locally to pull and save the image, and that your UbiOps subscription has access to custom environments.

## 1. Set up a connection with the UbiOps API client

First, we'll install the [UbiOps Python Client Library](https://ubiops.com/docs/python_client_library/) and initialize our connection to UbiOps.


```python
!pip install -qU ubiops requests
```

Now, we will need to initialize all the necessary variables for the UbiOps deployment. 

See [here](https://ubiops.com/docs/organizations/service-users/) to learn how you can get these variables.


```python
# Initialize variables
API_TOKEN = "<INSERT API TOKEN WITH PROJECT EDITOR RIGHTS>"
PROJECT_NAME = "<INSERT YOUR PROJECT NAME>"
API_HOST_URL = "<INSERT YOUR HOST API URL>"  # Standard UbiOps API URL is 'https://api.ubiops.com/v2.1'

DEPLOYMENT_NAME = "docling-serve"
DEPLOYMENT_VERSION = "v1"
SERVICE_NAME = "docling-service"
ENVIRONMENT_NAME = "docling-environment"

print(f"Your deployment will be named: {DEPLOYMENT_NAME}")
print(f"Your service will be named: {SERVICE_NAME}")
print(f"Your environment will be named: {ENVIRONMENT_NAME}")
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

## 2. Building the Docling Serve Docker image

We'll build the [Docling Serve](https://github.com/docling-project/docling-serve) Docker image locally. Docling Serve is a FastAPI-based REST API server that provides document processing capabilities through HTTP endpoints. The Docker image contains all necessary dependencies including Docling, OCR engines, and document processing libraries.

### Pull the image and build

First, pull the docling serve image and build the Docker image. The Dockerfile is already configured with all necessary dependencies and the server entry point.


```python
# Pull the latest Docling Serve image
!docker pull quay.io/docling-project/docling-serve:latest
```

### Save the Docker image

After pulling, we need to save the Docker image as a compressed tar archive. This archive will be uploaded to UbiOps as a [custom environment](https://ubiops.com/docs/environments/#bring-your-own-docker-image) using our Docker image. The save process exports the entire image including all layers, which may take a few minutes depending on the image size.


```python
# Save the Docker image as a tar archive
!docker save quay.io/docling-project/docling-serve:latest -o docling-serve.tar.gz
print("Docker image saved as docling-serve.tar.gz")
```

## 3. Creating a custom environment in UbiOps

Now we'll create a [custom environment](https://ubiops.com/docs/environments/#bring-your-own-docker-image) in UbiOps and upload the Docling Serve Docker image to it. Custom environments allow you to bring your own Docker images with pre-configured software stacks, libraries, and runtime configurations. This is ideal for deploying containerized applications like Docling Serve that have specific dependencies and server configurations already built into the image.

### Create the environment

First, we create an environment resource in UbiOps. The environment acts as a container for our Docker image and can be reused across multiple deployments.


```python
# Create custom environment
environment = api.environments_create(
    project_name=PROJECT_NAME,
    data=ubiops.EnvironmentCreate(
        name=ENVIRONMENT_NAME,
        display_name="Docling Serve Environment",
        description="Custom environment with Docling Serve Docker image for document processing",
        supports_request_format=False  # Docker image handles its own request format
    )
)

print(f"Created environment: {environment.name}")
print(f"Environment ID: {environment.id}")
```

### Upload the Docker image

Now we upload the Docker image tar archive to the environment. This process may take several minutes depending on the image size and your internet connection speed. UbiOps will extract and prepare the image for use in deployments.


```python
# Upload Docker image to environment
upload_response = api.environment_revisions_file_upload(
    project_name=PROJECT_NAME,
    environment_name=ENVIRONMENT_NAME,
    file='./docling-serve.tar.gz'
)

print(f"Upload started. Revision ID: {upload_response.revision}")
print("Uploading Docker image... This may take several minutes.")
```


```python
# Wait for environment to be ready
print("Waiting for environment build to complete...")
ubiops.utils.wait_for_environment(
    client=api.api_client,
    project_name=PROJECT_NAME,
    environment_name=ENVIRONMENT_NAME
)
print("Environment is ready!")
```

## 4. Create UbiOps deployment

### Deployment Configuration

We create a [deployment](https://ubiops.com/docs/deployments/) with `supports_request_format: False` since the Docker image contains a complete server application that handles its own request/response format. The deployment serves as a container for running the Docling Serve application. When accessed through Services, requests go directly to the Docling server running inside the container.

We also configure [environment variables](https://ubiops.com/docs/environment-variables/) at the deployment level to redirect cache directories to writable locations. Docling's OCR engine (EasyOCR) and other dependencies attempt to download models to the home directory on first use, which is read-only in certain (OpenShift) UbiOps installation environments. By setting these environment variables, we redirect all cache and model downloads to `/tmp`, which is always writable.


```python
# Create deployment
deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data=ubiops.DeploymentCreate(
        name=DEPLOYMENT_NAME,
        description="Docling Serve document processing server",
        supports_request_format=False,
        labels={"type": "ocr", "framework": "docling"}
    )
)
print(f"Created deployment: {deployment.name}")
```

### Deployment Version Configuration

We create a [deployment version](https://ubiops.com/docs/deployments/deployment-versions/) using our custom environment. The key difference from a standard deployment is that we specify our custom environment name instead of a standard Python environment. We set `minimum_instances: 1` to keep at least one instance running since document processing benefits from having the models loaded and ready. We use a CPU instance since Docling can run efficiently on CPU for most document processing tasks, though GPU instances can be used for faster OCR processing but in this case a different image needs to be used that has GPU support, check the [Docling Serve GitHub](https://github.com/docling-project/docling-serve) page for more info on their distributed images.


```python
# Create deployment version with custom environment
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment=ENVIRONMENT_NAME,  # Use our custom environment
    instance_type_group_name="16384 MB + 4 vCPU",  
    minimum_instances=1,  # Keep at least one instance warm
    request_retention_mode="full"  # Store request logs
)

deployment_version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=version_template,
)

print(f"Created deployment version: {deployment_version.version}")
print(f"Using environment: {deployment_version.environment}")
```

### Creating environment variables

We configure [environment variables](https://ubiops.com/docs/environment-variables/) at the deployment version level for two purposes:

1. **Cache directories**: Redirect model downloads to writable locations. Docling's OCR engine (EasyOCR) and other dependencies attempt to download models to the home directory on first use, which is read-only in certain (OpenShift) UbiOps installation environments. By setting these variables, we redirect all cache and model downloads to `/tmp`, which is always writable.

2. **Enable Gradio UI**: Setting `DOCLING_SERVE_ENABLE_UI=1` enables the Docling Serve web interface at the `/ui` endpoint, which can be accessed using browser authentication (see Chapter 8 for the guide).


```python
# Create environment variables for cache directories
env_vars = [
    {"name": "EASYOCR_MODULE_PATH", "value": "/tmp/.EasyOCR"},
    {"name": "TORCH_HOME", "value": "/tmp/.torch"},
    {"name": "TRANSFORMERS_CACHE", "value": "/tmp/.transformers"},
    {"name": "HF_HOME", "value": "/tmp/.huggingface"},
    {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
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

It may take a few minutes until your instance is provisioned and initialized. You can check the UI for any progress.

## 5. Create a Service to expose the Docling API

### Understanding Services

Now we'll create a [Service](https://ubiops.com/docs/services/) that exposes our Docling Serve server to the internet. The Service connects to the port where Docling Serve is running (default is 5000 for the FastAPI server), provides a public HTTPS endpoint, automatically handles TLS certificates, and load balances across deployment replicas.

### Service Configuration

The service requires a name, the deployment to connect to, the specific version to use, and the port number which must match the port Docling Serve listens on. The [Docling Serve](https://github.com/docling-project/docling-serve) runs on port 5001 by default. After creation, the service will be accessible at `https://[service-id].services.ubiops.com`. This URL will route directly to your Docling server's API. We also specify the `health_check_path` to point to the `/health` endpoint so UbiOps can monitor service availability. For authentication, we configure the service to require a UbiOps API token passed in the `Authorization` header of requests. This ensures only authorized users can access the service, leveraging UbiOps' existing [permission system](https://ubiops.com/docs/services/authentication/) to control who can process documents. 


```python
# Create a Service
service = api.services_create(
    project_name=PROJECT_NAME,
    data=ubiops.ServiceCreate(
        name=SERVICE_NAME,
        deployment=DEPLOYMENT_NAME,
        version=DEPLOYMENT_VERSION,
        port=5001,  # Docling Serve default port
        health_check={"path": "/health"},
        authentication_required=True,  # Require UbiOps token authentication
    )
)
SERVICE_URL = f"https://{service.id}.services.ubiops.com"

print(f"Service created: {service.name}")
print(f"Service ID: {service.id}")
print(f"Service URL: {SERVICE_URL}")
print(f"Document conversion endpoint: {SERVICE_URL}/v1/convert/source")
print(f"API documentation: {SERVICE_URL}/docs")
```

## 6. Test the document processing service

Now let's test our Docling service by sending document files for processing. Docling Serve provides [REST API](https://www.ibm.com/think/topics/rest-apis#:~:text=level%20agreement%20(SLA)-,REST%20APIs%20defined,APIs%20or%20RESTful%20web%20APIs.) endpoints for converting documents to various formats.

### Test 1: Check server health


```python
import requests


# Test health endpoint with authentication
headers = {"Authorization": API_TOKEN}
health_response = requests.get(f"{SERVICE_URL}/health", headers=headers)
print(f"Server health: {health_response.status_code}")
if health_response.status_code == 200:
    print("Docling Serve is healthy!")
    print(health_response.json())
else:
    print(f"Response: {health_response.text}")
```

### Test 2: Convert a document to Markdown

Define a URL for conversion using the `/v1/convert/source` endpoint. The example below uses a test URL pointing to the Docling documentation PDF from arXiv. You can replace this URL with any publicly accessible document in formats such as PDF, DOCX, PPTX, HTML, or images.


```python
import requests
import json

request_data = {
    "sources": [
        {"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}
    ],
    "options": {
        "to_formats": ["md"]
    },
    "target": {"kind": "inbody"}
}

response = requests.post(
    f"{SERVICE_URL}/v1/convert/source",
    headers={**headers, "Content-Type": "application/json"},
    json=request_data,
    timeout=300
)

if response.status_code == 200:
    print("Conversion successful!")
    result = response.json()
    doc = result['document']
    
    print(f"\nFilename: {doc['filename']}")
    print(f"Available content keys: {list(doc.keys())}")
    

    print(f"\nMarkdown content length: {len(doc['md_content'])} characters")
    print(f"\nMarkdown preview (first 1000 chars):\n{doc['md_content'][:1000]}")
else:
    print(f"Error: {response.status_code}")
```

### Test 3: Convert document to JSON format


```python
import requests
import json

request_data = {
    "sources": [
        {"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}
    ],
    "options": {
        "to_formats": ["json"]
    },
    "target": {"kind": "inbody"}
}

response = requests.post(
    f"{SERVICE_URL}/v1/convert/source",
    headers={**headers, "Content-Type": "application/json"},
    json=request_data,
    timeout=300
)

if response.status_code == 200:
    print("Conversion successful!")
    result = response.json()
    doc = result['document']
    
    print(f"\nFilename: {doc['filename']}")
    print(f"Available content keys: {list(doc.keys())}")
    

    print(f"\nJSON content length: {len(doc['json_content'])} characters")
    print(f"\nJSON structure preview:\n{json.dumps(doc['json_content'], indent=2)[:1000]}")
else:
    print(f"Error: {response.status_code}")

```

### Test 4: Convert document to HTML and plain text


```python
request_data = {
    "sources": [
        {"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}
    ],
    "options": {
        "to_formats": ["html", "text"],
        "do_ocr": False
    },
    "target": {"kind": "inbody"}
}

response = requests.post(
    f"{SERVICE_URL}/v1/convert/source",
    headers={**headers, "Content-Type": "application/json"},
    json=request_data,
    timeout=300
)

if response.status_code == 200:
    print("Conversion successful!")
    doc = response.json()['document']
    print(f"\nFilename: {doc['filename']}")
    
    if 'html_content' in doc and doc['html_content']:
        print(f"\nHTML content length: {len(doc['html_content'])} characters")
        print(f"HTML preview:\n{doc['html_content'][:500]}")
    
    if 'text_content' in doc and doc['text_content']:
        print(f"\nText content length: {len(doc['text_content'])} characters")
        print(f"Text preview:\n{doc['text_content'][:500]}")
else:
    print(f"Error: {response.status_code}")
```

### Test 5: Convert a local document to markdown


```python
import os

# Replace with your local file path
file_path = "path_to_your_local_file"

if os.path.exists(file_path):
    with open(file_path, "rb") as file:
        files = {"files": file}
        data = {
            "options[to_formats]": "md", # markdown
            "options[do_ocr]": "true", 
            "target_type": "inbody"
        }
        
        response = requests.post(
            f"{SERVICE_URL}/v1/convert/file",
            headers={"Authorization": API_TOKEN},
            files=files,
            data=data,
            timeout=300
        )
    
    if response.status_code == 200:
        print("Conversion successful!")
        result = response.json()
        doc = result['document']
        
        print(f"\nFilename: {doc['filename']}")
        print(f"Available content keys: {list(doc.keys())}")
        

        print(f"\nMarkdown content length: {len(doc['md_content'])} characters")
        print(f"\nMarkdown preview (first 1000 chars):\n{doc['md_content'][:1000]}")
    else:
        print(f"Error: {response.status_code}")
```

## 7. Accessing endpoints like `/docs` or `/ui` from your browser

For accessing authenticated endpoints like `/docs` or `/ui` (we have enabled the UI playground by using environment variable `DOCLING_SERVE_ENABLE_UI=1`) in your browser, you can use browser extension tools such as [Requestly](https://requestly.com/) to automatically inject authentication headers into requests. These tool will allow you to configure rules that add your Authorization header to all requests going to your service URL, enabling interaction with the Swagger UI documentation.

### Accessing API Documentation with Browser Authentication

The Docling Serve API documentation is available at `/swagger`, `/docs`, and `/scalar` endpoints. However, since authentication is required at the UbiOps Service level, you'll need to inject the Authorization header using a browser extension.

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
- Scalar: `{SERVICE_URL}/scalar`
- Docling serve UI: `{SERVICE_URL}/ui` 


**Note:** The exact workflow for creating rules differs per browser and Requestly version. Refer to [Requestly's documentation](https://docs.requestly.com/general/getting-started/introduction) for browser-specific instructions.

## 8. Cleanup
When you're done testing, clean up your active instance to avoid charges. The statement below was added to prevent scaling down the instance when running the full notebook at once. Remove it or comment it out to scale down the instance to 0.


```python
raise SystemExit("Prevented running all cells to avoid scaling down the deployment before completing all chapters. Please execute the last cell manually to scale down the instance.")
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

We have set up a deployment that hosts a Docling Serve document processing server using a custom Docker image. This tutorial demonstrates how to deploy document processing applications on UbiOps using bring-your-own-Docker and Services. Feel free to reach out to our [support portal](https://www.support.ubiops.com) if you want to discuss your set-up in more detail.
