# Deploy Jupyter Server with UbiOps Services

In this tutorial, we will explain how to deploy a [Jupyter Server](https://jupyter.org/) on UbiOps. A Jupyter Server is a backend component that provides the core services, APIs, and REST endpoints for Jupyter web applications such as Jupyter Notebook and JupyterLab. It runs in the background and manages the creation, communication, and lifecycle of notebook kernels, which are the computational engines executing Python code in notebooks.

Jupyter Server, hosted on UbiOps, can primarily be used to experiment with code in the target runtime environment. This allows you to for example develop code that uses a GPU if you do not have easy access to a GPU in your development environmnent. 

Note that UbiOps does not support volume mounting. All data that you import or generate, is lost after scaling down your Jupyter server.

### 1. Setup the UbiOps client

First, let's install the required packages and set up authentication.


```python
!pip install ubiops requests -qU
```

Now, we need to initialize all the necessary variables for the UbiOps deployment. To generate the API token you can follow this [guide](https://ubiops.com/docs/organizations/service-users/).

Once you have an API token, paste it below before continuing. Also fill in the name of your UbiOps project.


```python
## Add the name of your project and your API token
API_TOKEN = "Token "  # Add your API token here
PROJECT_NAME = ""  # Add your project name here

## Set custom names if you want, please refrain from using underscores and spaces
DEPLOYMENT_NAME = "jupyter-server"
DEPLOYMENT_VERSION = "v1"
SERVICE_NAME = 'jupyter-service'

## Change the instance type group if needed
INSTANCE_TYPE = "4096 MB + 1 vCPU" # You can find all possible Instance type groups in the WebApp under Project Admin > Project settings > Instance type groups
API_HOST_URL = "https://api.ubiops.com/v2.1" # This is the current UbiOps SaaS API URL, this URL may change with future installations
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

### 2. Our Jupyter token
Before we can create our deployment we need to generate an access token for our Jupyter Server, this token will be used later on when to connect to our server. 


```python
import uuid

JUPYTER_TOKEN = str(uuid.uuid4())
print(JUPYTER_TOKEN)
```

### 3. Create a deployment package
In order to create the Jupyter server inside our deployment, we need to provide the packages and dependencies for the environment and create a deployment script that will run the Jupyter server. Any `apt` and `pip` dependencies can be installed by adding them to the files below.

To do this we first create a folder for our files.


```python
import os

dir_name = "deployment_package"
os.makedirs(dir_name, exist_ok=True)
```

Python packages that you want to install with pip can be added to the `requirements.txt` file.


```python
%%writefile {dir_name}/requirements.txt
jupyterlab==4.0.11
notebook==7.0.7
ipywidgets==8.1.2
```

Any OS packages that you might need, such as CUDA drivers, can be added to the YAML file below.


```python
%%writefile {dir_name}/ubiops.yaml
apt:
  packages:
    - ubuntu-standard
```

Lastly, our deployment code with our Jupyter token inserted. This script contains a `Deployment` class with two key methods:

- **`__init__` Method**  
  This method runs when the deployment starts. It will open a server on the port we specified. We will connect to this port with a service in order to expose the endpoints of the Jupyter server through UbiOps.

- **`request()` Method**  
  The request method contains the logic for processing incoming data. This method executes the calls that are being made to the REST API endpoints, but now, because the endpoints of Jupyter Server are exposed by Services, calls to the Jupyter Server API can be handled instead. This is the reason that the request function can remain empty here.

For a complete overview of the deployment code structure, refer to the [UbiOps documentation](https://ubiops.com/docs/deployments/deployment-package/deployment-structure/).


```python
deployment_script = f"""
import subprocess
import urllib.request

JUPYTER_TOKEN = "{JUPYTER_TOKEN}"

# This class allows us to have custom error messages when the deployment fails
class UbiOpsError(Exception):
    def __init__(self, error_message):
        super().__init__()
        self.public_error_message = error_message

class Deployment:
    def __init__(self):
        try:
            self.proc = subprocess.Popen(['jupyter', 'notebook', '--ip', '0.0.0.0', '--IdentityProvider.token', JUPYTER_TOKEN],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            raise UbiOpsError("Unable to start a Jupyter notebook, are the packages jupyterlab and notebook installed?")

        # Get the IP address and print to the logs
        http_request = urllib.request.urlopen("https://whatismyipv4.ubiops.com")
        ip_address = http_request.read().decode("utf8")
        http_request.close()

        self.notebook_url = f"http://{{ip_address}}:8888/tree?token={{JUPYTER_TOKEN}}"
        print(f"Notebook URL: {{self.notebook_url}}")

    def request(self, data):
        return {{"notebook_url": self.notebook_url}}

    def stop(self):
        # Stop the Jupyter environment when the deployment is shutting down
        if self.proc is not None:
            self.proc.kill()
"""

with open(f"{dir_name}/deployment.py", "w") as f:
    f.write(deployment_script)
```

We zip the contents of our deployment package folder so we can upload it to our deployment later on.


```python
import shutil

deployment_zip_path = shutil.make_archive(dir_name, 'zip', dir_name)
```

### 4. Creating the Deployment

We create a deployment on UbiOps. This deployment will host our Jupyter Lab service.


```python
## Create the deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description="Run a Jupyter server from a deployment",
    input_type="plain",
    output_type="plain",
    input_fields=[],
    output_fields=[],
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)
```

Next, we create a new version for our deployment. We will upload our deployment package to this version.

The minimum number of instances for this deployment is set to 1. This means that the deployment will directly activate an instance and continue to run this instance if it is not turned off. We will scale down the active instance at the end of this tutorial. 

**Note that when this instance is not turned off manually it will continuously consume resources and credits!**


```python
## Create a version of the deployment
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-12',
    instance_type_group_name=INSTANCE_TYPE,
    minimum_instances=1, # The deployment instance is continuously active when the minimum number of instances is set at 1.
    maximum_instances=1,
    maximum_idle_time=900,
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=version_template
)
```

Upload the .zip file of our deployment package to the deployment version we just created. 

Logs of the build can be found in the Web App under Deployment > Logs or by passing `stream_logs=True` in the `wait_for_deployment_version` function. 

See `help(ubiops.utils.wait_for_deployment_version)` for help.


```python
## Upload the deployment_package to the deployment
upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file=deployment_zip_path,
)
print(upload_response)

## Wait for the revised deployment to be finished building...
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=upload_response.revision,
    # stream_logs=True,
)
```

### 5. Creating the Service

Next, we will create a UbiOps Service that exposes the Jupyter Lab server running on port 8888. Services provide automatic SSL, DNS, and authentication for your HTTP server. 

Authorization should be provided by the Jupyter token that we created and through the API token of UbiOps.


```python
## Create the service configuration
service_template = ubiops.ServiceCreate(
    name=SERVICE_NAME,
    deployment=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    port=8888,
    authentication_required=True,
    rate_limit_token=300,
)

service = api.services_create(
    project_name=PROJECT_NAME,
    data=service_template
)

print(f"Service '{SERVICE_NAME}' created successfully")
```

### 6. Accessing the Jupyter server

Now that the service is running, you can access Jupyter Server through your browser or connect to it from your local Jupyter Lab installation.

Services provides a URL based on the **service_id** that is structured as follows: `service_id` + `services.ubiops.com`. The service ID and the full URL are printed below and can be found in the Web App under Services.

Open the URL in your browser to access the server directly.


```python
import requests

service = api.services_get(
    project_name=PROJECT_NAME,
    service_name=SERVICE_NAME
)

headers = {"Authorization": API_TOKEN}

service_id = service.id
SERVICE_URL = f"https://{service_id}.services.ubiops.com"

print("Your UbiOps authorization token is", API_TOKEN)
print("Your Jupyter authorization token is", JUPYTER_TOKEN)
print(f"The full URL is {SERVICE_URL}/tree?token={JUPYTER_TOKEN}", headers)

## Check if the server is online
try:
    response = requests.get(f"{SERVICE_URL}/tree?token={JUPYTER_TOKEN}")
    print(f"Service status code: {response.status_code}")
    if response.status_code == 200:
        print("Jupyter Server is running and accessible!")
    else:
        print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"Connection test error: {e}")
```

### 7. Using Jupyter Server via API

Jupyter Server provides a REST API for programmatic access. Here are some examples of how to interact with it.

We can list all available kernels.


```python
import requests

# List available kernels
response = requests.get(f"{SERVICE_URL}/api/kernelspecs?token={JUPYTER_TOKEN}", headers=headers)

if response.status_code == 200:
    print("Available kernels:")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

List the notebooks that are present.


```python
# List contents of the root directory
response = requests.get(f"{SERVICE_URL}/api/contents?token={JUPYTER_TOKEN}", headers=headers)

if response.status_code == 200:
    print("Contents:")
    for item in response.json()['content']:
        print(f"  {item['type']}: {item['name']}")
else:
    print(f"Error: {response.status_code}")
```

Create a new notebook.


```python
# Create a new notebook
response = requests.post(
    f"{SERVICE_URL}/api/contents?token={JUPYTER_TOKEN}",
    headers=headers,
    json={
        "type": "notebook",
        "name": "test_notebook.ipynb"
    }
)

if response.status_code == 201:
    print("Notebook created successfully!")
    print(f"Path: {response.json()['path']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### 8. Connect from Local JupyterLab

You can connect to the remote Jupyter Server from your local JupyterLab installation. This allows you to use UbiOps resources while working in your familiar local environment.

First, make sure you have JupyterLab installed locally. If not, install it with:
```bash
pip install jupyterlab
```


```python
gateway_url = SERVICE_URL.replace('https://', 'http://')

print("Run this command in your local terminal to connect JupyterLab to the remote server:")
print(f"\njupyter lab --gateway-url={gateway_url} --GatewayClient.auth_token={JUPYTER_TOKEN}")
print("\nThis will start a local JupyterLab interface that uses the UbiOps deployment resources.")
```

### 9. Accessing Jupyter Server endpoints from your browser

For accessing the Jupyter server endpoints in your browser, you can use browser extension tools such as [Requestly](https://requestly.com/) to automatically inject authentication headers into requests. These tool will allow you to configure rules that add your Authorization header to all requests going to your service URL, enabling interaction with all the endpoints found in the [Swagger UI documentation](https://petstore.swagger.io/?url=https://raw.githubusercontent.com/jupyter/jupyter_server/master/jupyter_server/services/api/api.yaml#/sessions/post_api_sessions). Since authentication is required at the UbiOps Service level, you will need to inject the Authorization header using a browser extension.

[Requestly](https://requestly.io/) is a browser extension (available for Chrome, Firefox, Edge) that allows you to modify HTTP headers for specific URLs.

**Steps:**
1. Install the Requestly extension for your browser.
2. Choose HTTP Interceptor > Modify headers.
3. Configure the rule so that it includes your service URL ('services.ubiops.com').
4. Add a Request Header and choose 'authorization', fill the Header Value with the UbiOps token in the form 'Token ...'.
5. Save the rule.
6. You can now navigate to the endpoints in your browser.

### 10. Cleanup
When you're done testing, clean up your active instance to avoid charges. The statement below was added to prevent scaling down the instance when running the full notebook at once. Remove it or comment it out to scale down the instance to 0.


```python
raise SystemExit("Prevented running all cells to avoid scaling down the deployment before completing all chapters. Please execute the last cell manually to scale down the instance.")
```


```python
template = ubiops.DeploymentVersionUpdate(
    minimum_instances=0,
)

deployment_version = api.deployment_versions_update(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=template,
)
```

The active instance has now been scaled down.

## Summary
In this tutorial, you learned how to:

1. **Deploy Jupyter Server** as a service for interactive development.
2. **Create a UbiOps Service** that exposes Jupyter Server on port 8888.
3. **Access Jupyter Server** connect from local JupyterLab.
4. **Interact with Jupyter Server** programmatically via its REST API.
5. **Setup Requestly** to automatically authorize in your browser.

## Resources
- [UbiOps WebApp](https://app.ubiops.com/)
- [UbiOps Services Documentation](https://ubiops.com/docs/services/)
- [Jupyter Server Documentation](https://jupyter-server.readthedocs.io/)
- [Jupyter Server REST API](https://jupyter-server.readthedocs.io/en/latest/developers/rest-api.html)
- [Swagger Documentation](https://petstore.swagger.io/?url=https://raw.githubusercontent.com/jupyter/jupyter_server/master/jupyter_server/services/api/api.yaml#/sessions/post_api_sessions)
- [Requestly](https://requestly.io/)
