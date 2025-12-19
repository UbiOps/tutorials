# Deploy Gemma 2B with streaming on UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/huggingface-gemma2b/huggingface-gemma2b/huggingface-gemma2b.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/huggingface-gemma2b/huggingface-gemma2b/huggingface-gemma2b.ipynb){ .md-button .md-button--secondary }

This tutorial will help you create a cloud-based inference API endpoint for the gemma-2-2b-it model, using UbiOps. The generated text can be streamed back to an end-user. Gemma-2-2b-it is a lightweight LLM developed by Google, that can run on a CPU-type instance (does not require a GPU). It was developed by Google, and is available through Huggingface.

Note that Gemma is a gated model, so you will need to have a valid [Huggingface token](https://huggingface.co/docs/hub/en/security-tokens) with sufficient permissions if you want to download the gemma-2-2b-it from Huggingface. You can apply for one in [the repository of the respective model](https://huggingface.co/google/gemma-2-2b-it). The model can also be uploaded to your UbiOps bucket, and downloaded from there.

In this tutorial we will walk you through.

1. Connecting with the UbiOps API client
2. Creating a deployment for the Gemma 2 2B model
3. Create a request and stream the response

## 1. Connecting with the UbiOps API client
To use the UbiOps API from our notebook, we need to install the UbiOps Python client library first


```python
%pip install -qU ubiops
```

To set up a connection with the UbiOps platform API we need the name of your UbiOps project and an API token with `project-editor` permissions. See our documentation on how to [create a token](https://ubiops.com/docs/organizations/service-users/#service-users-and-api-tokens).

Once you have your project name and API token, paste them below in the following cell before running.


```python
import ubiops
from datetime import datetime

DEPLOYMENT_NAME = f"gemma-2-{datetime.now().date()}"
DEPLOYMENT_VERSION = "v1"

# Define our tokens
API_TOKEN = "<API TOKEN>"  # Make sure this is in the format "Token token-code"
PROJECT_NAME = "<PROJECT_NAME>"  # Fill in your project name here
HF_TOKEN = "<HF_TOKEN>"  # Format: "hf_xyz"


# Initialize client library
configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key["Authorization"] = API_TOKEN

# Establish a connection
client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
print(api.service_status())
```

## 2. Creating a deployment for Gemma 2 2B

We will now set up our deployment that runs the Gemma-2-2b-it model with streaming capabilities. First we create our deployment 
package - a directory in which our deployment files are added.

The deployment code is added to a `deployment.py` file, which has a `Deployment` class and two methods:
- `__init__` will run when the deployment starts up and can be used to load models, data, artifacts and other requirements 
for inference.
- `request()` will run every time a call is made to the model REST API endpoint and includes all the logic for processing 
data.

Additionally, we will add the dependencies that our code requires to a deployment package  a `requirements.txt`.


```python
deployment_package_dir = "deployment_package"

!mkdir {deployment_package_dir}
```

And add the `deployment.py` to the directory


```python
%%writefile {deployment_package_dir}/deployment.py
import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread


class Deployment:

    def __init__(self, base_directory, context):

        # Log in to Hugging Face
        token = os.environ["HF_TOKEN"]
        login(token=token)

        # Download Gemma from Hugging Face
        model_id = os.environ.get("MODEL_ID", "google/gemma-2-2b-it")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

        # You can change the system prompt by adding an environment variable to your deployment (version)
        self.system_prompt = os.environ.get(
            "SYSTEM_PROMPT",
            "You are a friendly chatbot who always responds in the style of a pirate",
        )

    def request(self, data, context):

        user_prompt = data
        streaming_callback = context["streaming_update"]

        # Prepare the chat prompt with the system message and user input
        chat = [{"role": "user", "content": f"{self.system_prompt} \n {user_prompt}"}]
        print("Applied chat: \n", chat)

        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=256)
        
        # The TextIteratorStreamer requires a thread which we start here
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            # We use the streaming_callback from UbiOps to send partial updates
            streaming_callback(new_text)
            generated_text += new_text

        return generated_text
```

Create a dependency file


```python
%%writefile {deployment_package_dir}/requirements.txt
huggingface-hub
transformers==4.45.2
torch==2.4.0

```

### Create the deploymenta & deployment version

#### Create the deployment


```python
# Create the deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    input_type="structured",
    output_type="plain",
    input_fields=[{"name": "prompt", "data_type": "string"}],
)

api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
```

And let us add our Huggingface token as a secret environment variable to our deployment, so that all of our deployment versions
are authenticated to download the relevant model files from Huggingface.


```python
api_response = api.deployment_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=ubiops.EnvironmentVariableCreate(name="HF_TOKEN", value=HF_TOKEN, secret=True),
)
```

#### And a deployment version


```python
# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment="python3-12",
    instance_type_group_name="12288 MB + 3 vCPU",
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=600,  # = 10 minutes
    request_retention_mode="full",
)

api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name=DEPLOYMENT_NAME, data=version_template
)
```


```python
# And now we zip our code (deployment package) and push it to the version

import shutil

deployment_code_archive = shutil.make_archive(
    deployment_package_dir, "zip", deployment_package_dir
)

upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file=deployment_code_archive,
)
print(upload_response)
```

Check if the deployment is finished building. Your first iteration will take around 10 minutes because a new environment is built.
Consecutive deployment code iterations will take only a couple of seconds because the environment was already built.


```python
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    revision_id=upload_response.revision,
)
```

You can check out the new deployment and the building process in the 
[UbiOps WebApp](https://app.ubiops.com) in the meantime.

If you are not happy with the default `SYSTEM_PROMPT` that we provided, you can add your own system prompt here


```python
CUSTOM_SYSTEM_PROMPT = "You are a friendly chatbot who always responds in the style of a man with a mission"

api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=ubiops.EnvironmentVariableCreate(
        name="SYSTEM_PROMPT", value=CUSTOM_SYSTEM_PROMPT, secret=False
    ),
)
```

## 3. Create a request and stream the response

We can now send our first prompt to our Gemma LLM! On the first spin-up, the model will need to be downloaded from Huggingface,
resulting in a cold-start time of a couple of minutes for your deployment instance. Subsequent requests should be handled faster. 
You can check the UbiOps User Interface to see the status and logs of your deployment instance while it is spinning up.
Once your instance is ready, tokens are streamed as they are generated by the Gemma model. Do note that this model has a 
rather long inference time in general. 



```python
data = {
    "prompt": "I accidentally brought the Black Plague on my ship. How do I blame the crew?"
}

# Create a streaming deployment request

for item in ubiops.utils.stream_deployment_request(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    data=data,
    timeout=3600,
    full_response=False,
    ):
    print(item, end="")
```

So that's it! You now have your own on-demand, scalable Gemma 2 2B Instruct model running in the cloud, with a REST API that you can reach from anywhere!
