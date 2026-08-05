# Implement PDF ingestion RAG pipeline with DSPy and UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/dspy-pipeline-tutorial/dspy-pipeline-tutorial){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/dspy-pipeline-tutorial/dspy-pipeline-tutorial/dspy-pipeline-tutorial.ipynb){ .md-button .md-button--secondary }

Note: This notebook runs on Python 3.12 and uses UbiOps 4.7.0

This notebook shows you how you can implement a PDF ingestion RAG pipeline for your LLM using the pipeline
functionality of UbiOps, DSPy, Langchain, PyPDF and Chroma. RAG is a framework that retrieves relevant or supporting context, and
adds them to the input. The original query and additional context are then fed to the LLM which produces the final output.
To create the proper prompt, we will be using [DSPy](https://dspy.ai/), a framework for programming - rather than prompting - language models.
For this tutorial, we will furthermore be using [PyPDF](https://pypdf.readthedocs.io/en/stable/) for the document ingestion and [Chroma](https://www.trychroma.com/) to store the embeddings of the PDF contents. With the pipeline we will build, you can upload a PDF and ask an LLM questions about the contents.
We will not deploy the LLM model in this tutorial; instead, we will use a mock-up. This approach eliminates the dependency on GPU usage, enabling you to run the tutorial entirely on CPU deployments.

Do furthermore note that it is preferable to run an independent vector store instead of running Chroma inside a deployment.
UbiOps deployments are inherently stateless, making it difficult to maintain a persistent vector store.
Just for this tutorial,
we will use Chroma inside the deployment for simplicity,
but be aware that this is not a recommended practice for production environments.

The framework will be set-up in a pipeline that contains three deployments: one that ingests and searches through a provided PDF file, another that creates a RAG enhanced prompt with the found information,
and one deployment with a mock-up LLM model. This split will allow for easy iteration and testing of the different components.

This framework can be set up in your UbiOps environment in five steps:
1) Establish a connection with your UbiOps environment
2) Create the deployment for the RAG from PDF
3) Create the deployment that creates the complete prompt for the LLM
4) Create the (mock) deployment for the LLM
5) Create a pipeline that combines the three deployments created in step 2,3 and 4

For this tutorial the [environments](https://ubiops.com/docs/environments/) will be created implicitly. This means that we
will create in total three deployment packages. which could contain these three files:
- `deployment.py`, the code that runs when a request is made (i.e., the embedding model & LLM model). This file is mandatory.
- `requirements.txt` and a `ubiops.yaml`, which will contain additional dependencies that UbiOps will add to the base environment. These files are optional.




## 1) Connecting with the UbiOps API client

To use the UbiOps API from our notebook, we need to install the UbiOps Python client library.


```python
!pip install --upgrade ubiops
```

To set up a connection with the UbiOps platform API we need the name of your UbiOps project and an API token with `project-editor` permissions.

Once you have your project name and API token, paste them below in the following cell before running.


```python
import ubiops
import shutil
import os

API_TOKEN = "Token ..." # Make sure this is in the format "Token token-code"
PROJECT_NAME = "..."  # Fill in your project name here

configuration = ubiops.Configuration()
configuration.api_key["Authorization"] = API_TOKEN

api_client = ubiops.ApiClient(configuration)
api = ubiops.api.CoreApi(api_client)
```

## Create the deployments for the pipeline

Now that we have established a connection with our UbiOps environment, we can start creating our deployment packages. Each
package could consist of three files (if necessary):
- The `deployment.py`, which is where we will define the actual Python code
- The `requirements.txt` and a `ubiops.yaml`, which will contain additional dependencies that our codes needs to run properly

These deployment packages will be zipped, and uploaded to UbiOps, after which we will add them to a pipeline. The pipeline
will consist out of three deployments:
- One deployment will be able to ingest a PDF and retrieve information from it
- One deployment that will create the prompt for the LLM
- One will host the (mock-up) LLM


```python
EMBEDDING_DEPLOYMENT_NAME = "rag-context-from-pdf"
LLM_DEPLOYMENT_NAME = "mock-up-llm"
PROMPT_DEPLOYMENT_NAME = "prompt"

deployment_directory = "deployments"

embedding_deployment_directory_name = "rag-context-from-pdf-deployment"
llm_deployment_directory_name = "mock-up-llm-deployment"
prompt_deployment_directory_name = "prompt-deployment"

embedding_deployment_directory_path = os.path.join(deployment_directory, embedding_deployment_directory_name)
llm_deployment_directory_path = os.path.join(deployment_directory, llm_deployment_directory_name)
prompt_deployment_directory_path = os.path.join(deployment_directory, prompt_deployment_directory_name)

```

### 2) Create the RAG from pdf deployment (Embedding)

The first deployment we will be creating is the one with the embedding model. This deployment will extract relevant information from the provided pdf. The retrieved information will be used as context in the final prompt sent to the LLM.


```python
os.makedirs(embedding_deployment_directory_path, exist_ok=True)
```

First we create the `deployment.py`:


```python
%%writefile {embedding_deployment_directory_path}/deployment.py
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

class Deployment:

    def __init__(self):
        print("Initializing deployment")
        self.embedding_model = os.getenv("embedding_model","sentence-transformers/all-MiniLM-l6-v2")

    def request(self, data):
        print("Processing request")
        context_file = data["context_file"]
        query = data["query"]

        loader = PyPDFLoader(context_file)

        print("Cutting document into chunks")
        docs_before_split = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 700,
            chunk_overlap  = 50,
        )
        docs_after_split = text_splitter.split_documents(docs_before_split)

        print("Loading embedding model from huggingface")
        huggingface_embedding_model = HuggingFaceBgeEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("Creating in-memory vector DB")
        vectorstore = Chroma.from_documents(docs_after_split, huggingface_embedding_model)

        print("retrieving relevant chunks from provided document")
        # Use similarity searching algorithm and return 3 most relevant documents.
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        context = [element.page_content for element in retriever.invoke(query)]
        context_string = '\n'.join(context)
        return {"context_string": context_string}
```

Then we will create the `requirements.txt` and `ubiops.yaml` to specify the required additional dependencies for the code above to run properly.


```python
%%writefile {embedding_deployment_directory_path}/requirements.txt
langchain==0.3.9
langchain-chroma==0.1.4
langchain-community==0.3.9
langchain-core==0.3.21
langchain-text-splitters==0.3.2
pypdf==5.1.0
sentence-transformers==3.3.1
```


```python
%%writefile {embedding_deployment_directory_path}/ubiops.yaml
apt:
  packages:
    - build-essential
```

#### Now we create the deployment

For the deployment we will specify the in- and output for the model:


```python
embed_template = ubiops.DeploymentCreate(
    name=EMBEDDING_DEPLOYMENT_NAME,
    description="A deployment that extracts relevant information from a provided pdf and returns this as a string",
    input_type="structured",
    output_type="structured",
    input_fields=[
        {"name": "query", "data_type": "string"},
        {"name": "context_file", "data_type": "file"}
    ],
    output_fields=[
        {"name": "context_string", "data_type": "string"}
    ],
    labels={"control": "embedding"},
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
    environment="python3-12",
    instance_type_group_name="8192 MB + 2 vCPU",
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=600,  # = 10 minutes
    request_retention_mode="full",  # input/output of requests will be stored
    request_retention_time=3600,  # requests will be stored for 1 hour
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name=EMBEDDING_DEPLOYMENT_NAME, data=version_template
)
print(version)
```

Then we zip the `deployment package` and upload it to UbiOps (this process can take around 10 minutes).


```python
import shutil

name = shutil.make_archive(embedding_deployment_directory_path, "zip", root_dir=deployment_directory, base_dir=embedding_deployment_directory_name)

file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=EMBEDDING_DEPLOYMENT_NAME,
    version="v1",
    file=name,
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=EMBEDDING_DEPLOYMENT_NAME,
    version="v1",
    revision_id=file_upload_result.revision,
)
```

### 3) Create the prompt node

The second deployment we will be creating is the one that creates the prompt for the LLM. This deployment will create a prompt that includes the original query and the context retrieved from the PDF.
The output will conform to the OpenAI format.
The output will contain a `body` dictionary and a `headers` dictionary, making it easy to use in the next step to send a request to an LLM accepting the OpenAI format.
By making use of the DSPy framework, we can easily create the prompt for the LLM.
Let's start by creating the `deployment.py`:


```python
os.makedirs(prompt_deployment_directory_path, exist_ok=True)
```


```python
%%writefile {prompt_deployment_directory_path}/deployment.py
import dspy
import httpretty
import json
import logging

from litellm import APIError


class Deployment:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        url = "http://mocked-api"
        self.lm = dspy.LM(
            model="openai/meta-llama/Llama-3.2-3B-Instruct",
            api_base=url,
            api_key="token ..."
        )
        dspy.configure(lm=self.lm)
        self.rag_prompt = dspy.Predict("context, question -> answer")
        self.intercepted_body = None
        self.intercepted_header = None
        self.register_uri = f"{url}/chat/completions"

        httpretty.register_uri(
            httpretty.POST,
            self.register_uri,
            body=self.mock_handler,
            content_type="application/json"
        )

    def mock_handler(self, request, uri, response_headers):
        logging.info(f"Intercepted request: {request.body or 'No body detected.'}")
        try:
            self.intercepted_body = json.loads(request.body.decode("utf-8")) if request.body else {}
            self.intercepted_header = dict(request.headers) if request.headers else {}
        except Exception as e:
            logging.error(f"Error processing intercepted request: {e}")
        return [200, response_headers, json.dumps({})]

    def request(self, data):
        with self.httpretty_context():
            try:
                logging.info("Triggering self.rag_prompt...")
                self.rag_prompt(context=data["context"], question=data["question"])
            except APIError:
                logging.warning("Request failed or intercepted.")

            assert self.intercepted_body is not None, "No body intercepted!"
            assert self.intercepted_header is not None, "No headers intercepted!"

            return {
                "body": self.intercepted_body,
                "headers": self.intercepted_header,
            }

    def httpretty_context(self):
        class HTTPrettyContext:
            def __enter__(self):
                httpretty.enable()

            def __exit__(self, exc_type, exc_val, exc_tb):
                httpretty.disable()
                httpretty.reset()

        return HTTPrettyContext()

```

Let's create the `requirements.txt` to install the required Python packages.
We will not need a `ubiops.yaml` file for this deployment, as no extra OS-level dependencies are required.


```python
%%writefile {prompt_deployment_directory_path}/requirements.txt
dspy
httpretty
```

#### Now we create the deployment

For the deployment we will specify the in- and output for the model:


```python
prompt_template = ubiops.DeploymentCreate(
    name=PROMPT_DEPLOYMENT_NAME,
    description="A deployment that creates a prompt for the LLM model",
    input_type="structured",
    output_type="structured",
    input_fields=[
        {"name": "question", "data_type": "string"},
        {"name": "context", "data_type": "string"}
    ],
    output_fields=[
        {"name": "body", "data_type": "dict"},
        {"name": "headers", "data_type": "dict"}
    ],
    labels={"control": "prompt"},
)

llm_deployment = api.deployments_create(project_name=PROJECT_NAME, data=prompt_template)
print(llm_deployment)
```

#### And finally we create the version

Each deployment can have multiple versions. The version of a deployment defines the coding environment, instance type (CPU or GPU)
& size, and other settings:


```python
version_template = ubiops.DeploymentVersionCreate(
    version="v1",
    environment="python3-12",
    instance_type_group_name="4096 MB + 1 vCPU",
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=600,  # = 10 minutes
    request_retention_mode="full",  # input/output of requests will be stored
    request_retention_time=3600,  # requests will be stored for 1 hour
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name=PROMPT_DEPLOYMENT_NAME, data=version_template
)
print(version)
```

Then we zip the `deployment package` and upload it to UbiOps (this process can take between 5-10 minutes).


```python
import shutil

name = shutil.make_archive(prompt_deployment_directory_path, "zip", root_dir=deployment_directory, base_dir=prompt_deployment_directory_name)

file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=PROMPT_DEPLOYMENT_NAME,
    version="v1",
    file=name,
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=PROMPT_DEPLOYMENT_NAME,
    version="v1",
    revision_id=file_upload_result.revision,
)
```

### 4) Create the LLM inference node

At last, we will create a mock-up LLM deployment. This deployment will mock the LLM model, and return a fixed response.
Since the output of our previous deployment is conforming to the OpenAI format, it will be easy to change this deployment to use an actual LLM model.
Let's start by creating the `deployment.py`:


```python
os.makedirs(llm_deployment_directory_path, exist_ok=True)
```


```python
%%writefile {llm_deployment_directory_path}/deployment.py

class Deployment:
    def __init__(self):
        pass

    def request(self, data):
        # Extract "body" and "headers" from the dictionary
        body = data.get("body", {})
        headers = data.get("headers", {})

        # Extract and format body content
        max_tokens = body.get("max_tokens", "N/A")
        model = body.get("model", "N/A")
        temperature = body.get("temperature", "N/A")

        messages = body.get("messages", [])
        formatted_messages = "\n".join(
            f"- Role: {message.get('role', 'N/A')}\n  Content:\n  {message.get('content', 'N/A')}"
            for message in messages
        )

        # Extract and format headers content
        formatted_headers = "\n".join(f"{key}: {value}" for key, value in headers.items())

        # Combine all parts into a formatted string
        formatted_output = (
            "=== Body ===\n"
            f"Model: {model}\n"
            f"Max Tokens: {max_tokens}\n"
            f"Temperature: {temperature}\n"
            f"Messages:\n{formatted_messages}\n\n"
            "=== Headers ===\n"
            f"{formatted_headers}"
        )

        return {"output": formatted_output}
```

#### Now we create the deployment

For the deployment we will specify the in- and output for the model:


```python
llm_mock_template = ubiops.DeploymentCreate(
    name=LLM_DEPLOYMENT_NAME,
    description="A deployment that creates a mock LLM output",
    input_type="structured",
    output_type="structured",
    input_fields=[
        {"name": "body", "data_type": "dict"},
        {"name": "headers", "data_type": "dict"}
    ],
    output_fields=[
        {"name": "output", "data_type": "string"}
    ],
    labels={"control": "llm_mock"},
)

llm_deployment = api.deployments_create(project_name=PROJECT_NAME, data=llm_mock_template)
print(llm_deployment)
```

#### And finally we create the version

Each deployment can have multiple versions. The version of a deployment defines the coding environment, instance type (CPU or GPU)
& size, and other settings:


```python
version_template = ubiops.DeploymentVersionCreate(
    version="v1",
    environment="python3-12",
    instance_type_group_name="512 MB + 0.125 vCPU",
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=600,  # = 10 minutes
    request_retention_mode="full",  # input/output of requests will be stored
    request_retention_time=3600,  # requests will be stored for 1 hour
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME, deployment_name=LLM_DEPLOYMENT_NAME, data=version_template
)
print(version)
```

Then we zip the `deployment package` and upload it to UbiOps (this process can take between 5-10 minutes).


```python
import shutil

name = shutil.make_archive(llm_deployment_directory_path, "zip", root_dir=deployment_directory, base_dir=llm_deployment_directory_name)

file_upload_result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=LLM_DEPLOYMENT_NAME,
    version="v1",
    file=name,
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=LLM_DEPLOYMENT_NAME,
    version="v1",
    revision_id=file_upload_result.revision,
)
```

All of our deployments are now ready and uploaded to UbiOps. We can now create a pipeline that combines the three deployments.

### 5) Create the pipeline

Now we create a pipeline that orchastrates the workflow between the deployments above. When a request will be made to this pipeline
the first deployment will ingest the provided PDF and will search for information that's relevant to the user's query. This information will then be sent to the prompt deployment which will create a prompt out of the query and context.
 This prompt will then be sent to the LLM (mock) deployment, which will answer the user's query based on the provided information from the PDF.
 The pipeline will be as follows:



![pipeline-image](https://storage.googleapis.com/ubiops/tutorial-helper-files/dspy-pipeline-tutorial/DSPY-tutorial-pipeline-image.png)

For a pipeline you will have to define an input & output and create a version, as with a deployment. In addition to this we
will also need to define the objects (i.e, deployments) and how to orchestrate the workflow (i.e., how to attach each object
to each other).

First we create the pipeline:


```python
PIPELINE_NAME = "llama-with-rag"
PIPELINE_VERSION = "v1"
```


```python
pipeline_template = ubiops.PipelineCreate(
    name=PIPELINE_NAME,
    description="A pipeline to prepare prompts, and generate text using llama 3.3",
    input_type="structured",
    input_fields=[
        {"name": "query", "data_type": "string"},
        {"name": "context_file", "data_type": "file"}
    ],
    output_type="structured",
    output_fields=[
        {"name": "output", "data_type": "string"}
    ],
    labels={"demo": "llama-3-3-RAG"},
)

api.pipelines_create(project_name=PROJECT_NAME, data=pipeline_template)
```

Then we define the objects, and how to attach the objects together:


```python
objects = [
    # RAG
    {
        "name": EMBEDDING_DEPLOYMENT_NAME,
        "reference_name": EMBEDDING_DEPLOYMENT_NAME,
        "version": "v1",
    },
    # PROMPT
    {
        "name": PROMPT_DEPLOYMENT_NAME,
        "reference_name": PROMPT_DEPLOYMENT_NAME,
        "version": "v1"
    },
    # LLM-model
    {
        "name": LLM_DEPLOYMENT_NAME,
        "reference_name": LLM_DEPLOYMENT_NAME,
        "version": "v1"
    }
]

attachments = [
    # start --> RAG
    {
        "destination_name": EMBEDDING_DEPLOYMENT_NAME,
        "sources": [
            {
                "source_name": "pipeline_start",
                "mapping": [
                    {
                        "source_field_name": "query",
                        "destination_field_name": "query",
                    },
                    {
                        "source_field_name": "context_file",
                        "destination_field_name": "context_file",
                    }
                ]
            }
        ]
    },
    # RAG + pipeline-start -> PROMPT
    {
        "destination_name": PROMPT_DEPLOYMENT_NAME,
        "sources": [
            {
                "source_name": EMBEDDING_DEPLOYMENT_NAME,
                "mapping": [
                    {
                        "source_field_name": "context_string",
                        "destination_field_name": "context",
                    }
                ]
            },
            {
                "source_name": "pipeline_start",
                "mapping": [
                    {
                        "source_field_name": "query",
                        "destination_field_name": "question",
                    }
                ]
            }
        ]
    },
    # PROMPT -> LLM
    {
        "destination_name": LLM_DEPLOYMENT_NAME,
        "sources": [
            {
                "source_name": PROMPT_DEPLOYMENT_NAME,
                "mapping": [
                    {
                        "source_field_name": "body",
                        "destination_field_name": "body",
                    },
                    {
                        "source_field_name": "headers",
                        "destination_field_name": "headers",
                    }
                ]
            }
        ]
    },
    # LLm -> pipeline end
    {
        "destination_name": "pipeline_end",
        "sources": [
            {
                "source_name": LLM_DEPLOYMENT_NAME,
                "mapping": [
                    {
                        "source_field_name": "output",
                        "destination_field_name": "output",
                    }
                ]
            }
        ]
    }
]
```


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

That is all you need to know about how to set-up a RAG from file framework in UbiOps, using Langchain, Chroma and PyPDF.

Let's test it out by providing our pipeline with Nike's annual public SEC report and asking about its contents!


```python
from ubiops.utils import upload_file
from urllib.request import urlretrieve

url = "https://s1.q4cdn.com/806093406/files/doc_downloads/2023/414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.pdf"
location, _ = urlretrieve(url, "nike_sec.pdf")

file_uri= upload_file(
    client=api_client,
    project_name=PROJECT_NAME,
    file_path=location
)
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=PIPELINE_NAME,
    data={
        "query": "What was Nike's revenue in 2023?",
        "context_file": file_uri
    },
)
```

As you can see, the pipeline is able to retrieve relevant information from the PDF, create a prompt for the LLM, and return a (mock) answer to the user's query.
