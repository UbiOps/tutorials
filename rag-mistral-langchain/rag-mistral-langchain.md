# Implement RAG with Langchain on UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/rag-mistral-langchain/rag-mistral-langchain/rag-mistral-langchain.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/rag-mistral-langchain/rag-mistral-langchain/rag-mistral-langchain.ipynb){ .md-button .md-button--secondary }

Note: This notebook runs on Python 3.11 and uses UbiOps 4.1.0

This notebook shows you how you can implement a Retrieval-Augmented Generation (RAG) framework for your LLM using the pipeline
functionallity of UbiOps and Langchain. RAG is a framework that retrieves relevant or supporting context, and 
adds them to the input. The input and additional documents are then fed to the LLM which, produces the final output. For this 
tutorial we will be hosting the [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) LLM, 
use the embeddings from Cohere, and Pinecone to store these embeddings. The set-up we will create in this tutorial will 
give the user better recommandations of where to travel too.

The framework will be set-up in a pipeline that contains two deployments: one that hosts & searchers through an embedding
database (the instructor depoyment) that will be used to concatenate the user's prompt with additional embedding, 
and one deployment for where the LLM will be run.

This framework can be set up in your UbiOps environment in four steps:
1) Establish a connection with your UbiOps environment
2) Create the deployment for the embeddings
3) Create the deployment for the LLM
4) Create a pipeline that combines the two deployments created in step 2 and 3

**NOTE:** In order to complete this tutorial you will need an API key from [Cohere](https://dashboard.cohere.com/welcome/register)
 and [Pinecone](https://login.pinecone.io), you can acquire an API key after making an account for both platforms.
Since Mistral is behind a gated repository - you will also need to a Huggingface token that has sufficient permissions 
to download Mistral.

For this tutorial the [environments](https://ubiops.com/docs/environments/) will be created implicitly. This means that we 
will create two deployment packages. which will contain two files:
- `deployment.py`, the code that runs when a request is made (i.e., the embedding model & LLM model)
- `requirements.txt`,which will contain additional dependencies that UbiOps will add to the base environment




## 1) Connecting with the UbiOps API client

To use the UbiOps API from our notebook, we need to install the UbiOps Python client library.


```python
!pip install --upgrade ubiops
!pip install langchain
!pip install pinecone-client
!pip install cohere
```

To set up a connection with the UbiOps platform API we need the name of your UbiOps project and an API token with `project-editor` permissions.

Once you have your project name and API token, paste them below in the following cell before running.


```python
import ubiops
import shutil
import langchain
import os

API_TOKEN = "<UBIOPS_API_TOKEN>"  # Make sure this is in the format "Token token-code"
PROJECT_NAME = "<PROJECT_NAME>"  # Fill in your project name here

HF_TOKEN = "<ENTER YOUR HF TOKEN WITH ACCESS TO MISTRAL REPO HERE>"

configuration = ubiops.Configuration()
configuration.api_key["Authorization"] = API_TOKEN

api_client = ubiops.ApiClient(configuration)
api = ubiops.api.CoreApi(api_client)
```

Copy paste the API keys from Pinecone and Cohere below. We will turn these API keys into [environment variables](https://ubiops.com/docs/environment-variables/)
later on so we can access them from inside our deployment code we will define later.


```python
PINECONE_API_KEY = "<PINECONE_API_TOKEN>"
COHERE_API_KEY = "<COHERE_API_TOKEN>"
```


```python
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Pinecone
from langchain.docstore.document import Document
import cohere
import pinecone
import os

embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")

pinecone.create_index("ubiops-rag", dimension=4096)
docsearch = Pinecone.from_existing_index(index_name="ubiops-rag", embedding=embeddings)

new_doc = Document(page_content="description", metadata={"place": "location"})
inserted = docsearch.add_documents([new_doc])
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
EMBEDDING_DEPLOYMENT_NAME = "instructor"
LLM_DEPLOYMENT_NAME = "llm-mistral"
```

### 2) Create the Instructor node deployment (Embedding)

The first deployment we will be creating is the one with the embedding model. This deployment will instruct the LLM how to 
answer the question properly, and search for relevant places that will be added to the user prompt. Doing this will "guide"
the Mistral 7B model in the second deployment to a better answer. In order for the code inside the deployment to work properly
we will need to add the Pinecone and Cohere API tokens as environment variables. 


```python
!mkdir prompt_node
```

First we create the `deployment.py`:


```python
%%writefile prompt_node/deployment.py
from langchain.vectorstores import Pinecone
from langchain.embeddings import CohereEmbeddings
import pinecone
import os

class Deployment:

    def __init__(self, base_directory, context):
        print("Loading embeddings")
        embeddings = CohereEmbeddings(cohere_api_key=os.environ['COHERE_API_KEY'])
        pinecone.init(api_key=os.environ['PINECONE_API_KEY'],
              environment="gcp-starter")
        print("Searching through embeddings")
        self.docsearch = Pinecone.from_existing_index(index_name="ubiops-rag", embedding=embeddings).as_retriever()

        self.template = """
        <s> [INST]You are an expert in travelling around the world. A user asked you an advice for the trip. 
        Recommend him to go to {location}, also mention facts from following context. [/INST] </s> 
        [INST] Question: {question}
        Context: {context} 
        Recomendation: [/INST]
        """

    def request(self, data, context):

        question = data["request"]
        print("Processing request")
        place = self.docsearch.get_relevant_documents(question)[0]
        instruction = self.template.format(location=place.metadata['place'], context=place.page_content, question=question)
        return {"location": place.metadata['place'], "instruction": instruction}
```

Then we will create the `requirements.txt` so specify the required additional dependencies for the code above to run properly.


```python
%%writefile prompt_node/requirements.txt
langchain
pinecone-client
cohere
```

#### Now we create the deployment 

For the deployment we will specify the in- and output for the model:


```python
embed_template = ubiops.DeploymentCreate(
    name=EMBEDDING_DEPLOYMENT_NAME,
    description="A deployment to create prompts for mistral",
    input_type="structured",
    output_type="structured",
    input_fields=[
        {"name": "request", "data_type": "string"},
    ],
    output_fields=[
        {"name": "location", "data_type": "string"},
        {"name": "instruction", "data_type": "string"},
    ],
    labels={"controll": "prompt"},
)

llm_deployment = api.deployments_create(project_name=PROJECT_NAME, data=embed_template)
print(llm_deployment)
```

#### And finally we create the version

Each deployment can have multiple versions. The version of a deployment defines the coding environment, instance type (CPU or GPU) 
& size, and other settings:


```python
version_template = ubiops.DeploymentVersionCreate(
    version="v1",
    environment="python3-11",
    instance_type_group_name="256 MB + 0.0625 vCPU",
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=1800,  # = 30 minutes
    request_retention_mode="full",  # input/output of requests will be stored
    request_retention_time=3600,  # requests will be stored for 1 hour
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name="instructor", data=version_template
)
print(version)
```

As mentioned earlier, we need to turn the API keys from Cohere & Pinecone into environment variables so we can access
them from inside the deployment code. This is done in the code cell below:


```python
# Create an environment variable for the Pinecone API token
pinecone_api_key = ubiops.EnvironmentVariableCreate(
    name="PINECONE_API_KEY", value=PINECONE_API_KEY, secret=True
)

api.deployment_version_environment_variables_create(
    PROJECT_NAME,
    deployment_name=EMBEDDING_DEPLOYMENT_NAME,
    version="v1",
    data=pinecone_api_key,
)

# Create an environment variable for the Cohere API token
cohere_api_key = ubiops.EnvironmentVariableCreate(
    name="COHERE_API_KEY", value=COHERE_API_KEY, secret=True
)


api.deployment_version_environment_variables_create(
    PROJECT_NAME,
    deployment_name=EMBEDDING_DEPLOYMENT_NAME,
    version="v1",
    data=cohere_api_key,
)
```

Then we zip the `deployment package` and upload it to UbiOps (this process can take between 5-10 minutes). 


```python
import shutil

shutil.make_archive("prompt_node", "zip", ".", "prompt_node")

file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=EMBEDDING_DEPLOYMENT_NAME,
    version="v1",
    file="prompt_node.zip",
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=EMBEDDING_DEPLOYMENT_NAME,
    version="v1",
    revision_id=file_upload_result.revision,
)
```

### 3) Create the LLM node deployment

Next we will create the deployment that will contain the LLM itself. As mentioned before we will be making use of the 
Mistral 7B Instruct. The workflow for creating this deployment is the same as the deployment for the embeddings: first we 
will create a `deployment.py`, then a `requirements.txt`, then the deployment (specifying the models input & output), and finish
off with creating a version for this deployment.


```python
!mkdir llm_model
```

Create the `deployment.py`:


```python
%%writefile llm_model/deployment.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from huggingface_hub import login
import transformers

class Deployment:

    def __init__(self, base_directory, context):
        
        token=os.environ['HF_TOKEN']

        login(token=token)

        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        print("Loading model weights")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=False,
        )
        print("Loading model")
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        self.pipeline = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            return_full_text=False,
            max_new_tokens=500)

    def request(self, data, context):

        result = self.pipeline(data["prompt"])[0]["generated_text"]
        print("Processing request")
        return {"generated_text": result}
```

Then the `requirements.txt`:


```python
%%writefile llm_model/requirements.txt
transformers
torch
bitsandbytes
accelerate
scipy
huggingface_hub
```

#### Create a deployment 


```python
llm_template = ubiops.DeploymentCreate(
    name="llm-mistral",
    description="A deployment to run mistral",
    input_type="structured",
    output_type="structured",
    input_fields=[
        {"name": "prompt", "data_type": "string"},
    ],
    output_fields=[{"name": "generated_text", "data_type": "string"}],
    labels={"controll": "llm"},
)

llm_deployment = api.deployments_create(project_name=PROJECT_NAME, data=llm_template)
print(llm_deployment)
```

And a version for the deployment:


```python
version_template = ubiops.DeploymentVersionCreate(
    version="v1",
    environment="ubuntu22-04-python3-11-cuda11-7-1",
    instance_type_group_name="16384 MB + 4 vCPU + NVIDIA Tesla T4",
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=1800,  # = 30 minutes
    request_retention_mode="full",  # input/output of requests will be stored
    request_retention_time=3600,  # requests will be stored for 1 hour
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name="llm-mistral", data=version_template
)
print(version)
```

Zip & upload the files to UbiOps (this process can take between 5-10 minutes).


```python
import shutil

shutil.make_archive("llm_model", "zip", ".", "llm_model")

file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name="llm-mistral",
    version="v1",
    file="llm_model.zip",
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name="llm-mistral",
    version="v1",
    revision_id=file_upload_result.revision,
)
```

Now we need to create an environment variable for our Huggingface token, to be able to download the model from Huggingface.
Make sure that the token has access to the gated repo from MistralAI.


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name="llm-mistral",
    version="v1",
    data=ubiops.EnvironmentVariableCreate(name="HF_TOKEN", value=HF_TOKEN, secret=True),
)
```

## 4) Create a pipeline and pipeline version

Now we create a pipeline that orchastrates the workflow between the deployments above. When a request will be made to this pipeline
the first deployment will search for a location according to the user's prompt, and will search for additional documents about
this location. This information will then be send to the LLM which will generate text on why that location is worth visiting.

For a pipeline you will have to define an input & output and create a version, as with a deployment. In addition to this we
will also need to define the objects (i.e, deployments) and how to orchestrate the workflow (i.e., how to attach each object
 to each other).

First we create the pipeline:


```python
PIPELINE_NAME = "llm-generator"
PIPELINE_VERSION = "v1"
```


```python
pipeline_template = ubiops.PipelineCreate(
    name=PIPELINE_NAME,
    description="A pipeline to prepare prompts, and generate text using Mistral",
    input_type="structured",
    input_fields=[
        {"name": "request", "data_type": "string"},
    ],
    output_type="structured",
    output_fields=[
        {"name": "location", "data_type": "string"},
        {"name": "generated_text", "data_type": "string"},
    ],
    labels={"demo": "mistral-RAG"},
)

api.pipelines_create(project_name=PROJECT_NAME, data=pipeline_template)
```

Then we define the objects, and how to attach the objects together:


```python
objects = [
    # preprocessor
    {
        "name": EMBEDDING_DEPLOYMENT_NAME,
        "reference_name": "instructor",
        "version": "v1",
    },
    # LLM-model
    {"name": LLM_DEPLOYMENT_NAME, "reference_name": "llm-mistral", "version": "v1"},
]

attachments = [
    # start --> instruction-generator
    {
        "destination_name": "instructor",
        "sources": [
            {
                "source_name": "pipeline_start",
                "mapping": [
                    {
                        "source_field_name": "request",
                        "destination_field_name": "request",
                    }
                ],
            }
        ],
    },
    # instruction-generator --> LLM
    {
        "destination_name": "llm-mistral",
        "sources": [
            {
                "source_name": "instructor",
                "mapping": [
                    {
                        "source_field_name": "instruction",
                        "destination_field_name": "prompt",
                    }
                ],
            }
        ],
    },
    # LLm -> pipeline end, instruction-generator -> pipeline end
    {
        "destination_name": "pipeline_end",
        "sources": [
            {
                "source_name": "instructor",
                "mapping": [
                    {
                        "source_field_name": "location",
                        "destination_field_name": "location",
                    }
                ],
            },
            {
                "source_name": "llm-mistral",
                "mapping": [
                    {
                        "source_field_name": "generated_text",
                        "destination_field_name": "generated_text",
                    }
                ],
            },
        ],
    },
]
```

And finally we create a version for this pipeline:


```python
pipeline_template = ubiops.PipelineVersionCreate(
    version=PIPELINE_VERSION,
    request_retention_mode="full",
    objects=objects,
    attachments=attachments,
)

api.pipeline_versions_create(
    project_name=PROJECT_NAME, pipeline_name=PIPELINE_NAME, data=pipeline_template
)
```

## And there you have it!

That is all you need to know about how to set-up a RAG framework in UbiOps, using Langchain, Cohere, and Pinecone. If you  
want you can use the code block below to create a request to your newly created pipeline.


```python
response = api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=PIPELINE_NAME,
    data={"request": "A place in the Europe"},
)

print(response.result)
```
