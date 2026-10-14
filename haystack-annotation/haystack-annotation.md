# Data annotation pipeline with Haystack

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/haystack-annotation/haystack-annotation/){ .md-button .md-button--primary } [View source code :fontawesome-brands-github:](https://github.com/UbiOps/tutorials/blob/master/haystack-annotation/haystack-annotation/haystack-annotation.ipynb){ .md-button }.

In this tutorial we will create a document enrichment pipeline on UbiOps. This pipeline can read documents for a Google Cloud Storage bucket, summarize their contents and add the summary into the metadata of the document. The pipeline is created using the pipeline feature of UbiOps and employs Haystack and Docling  for certain steps.

With a UbiOps pipeline, users can create modular workflows by connecting multiple deployments with one-another, allowing for a series of separate data transformation in one run. Pipeline operators can also be included to process data in parallel, insert variables or functions, or raise errors where needed. For more information, you can take a look at the [documentation](https://ubiops.com/docs/pipelines/) and this [tutorial](https://ubiops.com/docs/ubiops_tutorials/pipeline-tutorial/pipeline-tutorial/).

[Haystack](https://haystack.deepset.ai/) is an open-source LLM-Ops pipeline framework. Users can create pipelines and deploy LLMs for things like document processing, internet searching, conversational AI, and retrieval-augmented generation. Haystack also supports an integration of [Docling](https://www.docling.ai/), an open-source toolkit for document processing, allowing to turn all sorts of document into standardized data.

The document enrichment pipeline that we will create can be employed to summarize and annotate documents. First, documents will be fetched, then Haystack is used to run Docling, which can convert various types of documents into JSON. After that, Haystack is used to send requests to an (interchangable) OpenAI-compatible Large Language Model, which writes a technical summary of the converted documents. The summary is then written into the metadata of the document and stored separately.

All the files that will be processed are made directly accessible to our pipeline by connecting a Google Cloud Storage bucket to a UbiOps bucket, allowing them to synchronize with each other. This means that the first step of this tutorial is creating a GCS bucket. Using a UbiOps bucket to upload files and processing them directly from there is also possible, however the use of a GCS bucket is advised as UbiOps is not a data management platform. The step of creating and connecting a GCS bucket can be skipped when using a UbiOps-managed bucket for storing your files.

The pipeline will consist of four separate elements. 
- The first deployment will list the documents of the GCS bucket that can be processed. 
- A [subrequest operator](https://ubiops.com/docs/pipelines/operators/#create-subrequests) will send individual files to the rest of the pipeline so that files are processed concurrently.
- The second deployment downloads the files and converts them into a JSON using `docling-haystack`.
- The third deployment sends a request to a `GPT-4o mini` LLM with `haystack-ai`, which is tasked with summarizing the incoming JSON string.
- The last deployment writes the summary into the metadata of the original document and stores it separately.

### Configuration and setting up the UbiOps client
 
The first things that we will do are:
- Define configuration variables
- Install the `ubiops` library
- Initialize the UbiOps client

Regarding the configuration variables, to generate the `API_TOKEN` you can follow [this guide](https://ubiops.com/docs/organizations/service-users/). To ensure that you can create and update the deployments of this pipeline, make sure to grant the API token has `project-admin` or `project-editor` permissions.

Besides the API token we also need to insert the name of your UbiOps project and set a `BUCKET_NAME`. If you only want to process files from a specific location inside your bucket you can set the `BUCKET_PREFIX` to map to this location in the bucket.

An OpenAI key is needed for authentication when running the `GPT-4o mini`. You create a personal key by creating an OpenAI account and navigating to the [API key](https://platform.openai.com/settings/organization/api-keys) tab. Paste the secret key below in the `OPENAI_TOKEN` field. 

Instance types and deployment names will be defined separately for each of the elements in the pipeline.


```python
API_TOKEN = "<UBIOPS_API_TOKEN>"  ## Used to create the deployments and pipeline, make sure this is in the format "Token token-code"
API_HOST_URL = "<API_HOST_URL>"  ## Standard UbiOps API URL is 'https://api.ubiops.com/v2.1', your URL may differ depending on your environment
PROJECT_NAME = "<PROJECT_NAME>"  ## Fill in your project name here

BUCKET_NAME = "<BUCKET_NAME>"  ## Choose a name for your UbiOps bucket, this does not have to match with the name of the GCS bucket
BUCKET_PREFIX = "<BUCKET_PREFIX_FOLDER>"  ## When you only want to process files from a specific folder in your bucket add that path here

OPENAI_TOKEN = "<OPENAI_API_KEY>"  ## Fill in the OpenAI token here, it is in the format "sk-XXXXX..."
```

Install the `ubiops` library.


```python
!pip install -qU ubiops
```

Now we can do some imports and initialize the UbiOps client.


```python
import os
import shutil
import ubiops

configuration = ubiops.Configuration(host=API_HOST_URL)
configuration.api_key['Authorization'] = API_TOKEN
client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)

status_check = api.service_status()

print(status_check)
print(f"Connected to UbiOps project '{PROJECT_NAME}'")
```

### Creating a Google Cloud Storage bucket

After connecting to the client we can connect a GCS bucket to a bucket on UbiOps. To do this we need the following information of our GCS bucket:
- Name of the bucket
- Service account with *Storage Object Admin* permission
- JSON key file

You can create a new GCS bucket on [Google Cloud Storage](https://console.cloud.google.com/storage/browser/bucket) by going to the Buckets tab and clicking on Create. The name and the settings of the bucket can be chosen as preferred. After creating the bucket, the documents that you want to annotate can be added. For the purpose of this tutorial, it is recommended to not add a very large number of files. Let's start with a test PDF document. You can provide your own document(s), but if you need a few test documents you can download these famous scientific papers:
- ['Protein Measurement with the folin phenol reagent', 1951, Lowry et al.](https://developmentalbiology.wustl.edu/app/uploads/2018/10/Lowry-1951-2fwrw0a.pdf)
- ['Density-functional thermochemistry. III. The role of exact exchange', 1992, Becke](https://gwern.net/doc/biology/1993-becke.pdf)
- ['Simulating Physics with Computers', 1981, Feynman](https://s2.smu.edu/~mitch/class/5395/papers/feynman-quantum-1981.pdf)

The next step is creating a service account that has permission to access the bucket, and to upload, download and delete files. You can create a service account in Google Cloud by going to IAM & Admin > Service accounts > Create service account. It is important to grant the service account full read and write access by selecting the *Storage Object Admin* role. After creating the service account and granting it the correct permissions you need to download the credentials of the service account in JSON format. This can be done by navigating to IAM & Admin > Service accounts > Choose your new service accounts > Manage permissions > Keys > Add new key > JSON. Now the new JSON key will be saved to your machine. Make sure to store it safely.

Additional information on how to connect GCS buckets with UbiOps can be found [here](https://ubiops.com/docs/howto/buckets/howto-google-cloud-storage/).

Add the name of your GCS bucket and the path to your JSON key file below before continuing.


```python
GCS_BUCKET_NAME = "<GCS_BUCKET_NAME>"  ## This must be the name of your GCS bucket that you synced with UbiOps
JSON_KEY_FILE_PATH = "/path/to/local/json/key/file"  ## Path to the downloaded JSON key file
```

Now we can make a call to the UbiOps API to create a bucket that connects to our GCS bucket. For this we provide the name of our UbiOps bucket with `BUCKET_NAME` and the name of our GCS bucket with `GCS_BUCKET_NAME`. Our JSON key file is read into the credentials for authentication. A prefix can be inserted to select a specific folder/location in your GCS bucket, when it is left empty it sets to `/` and selects all files. When you select a folder with `BUCKET_PREFIX`, only this folder will be synced with the UbiOps bucket. The `ttl` is the 'time to live' in seconds, which must be a multiple of a week or None for unlimited. A description and labels can also be added.


```python
json_key_file = open(JSON_KEY_FILE_PATH).read()

bucket_data = {
    "name": BUCKET_NAME,
    "provider": "google_cloud_storage",
    "credentials": {
        "json_key_file": json_key_file
        },
    "configuration": {
        "bucket": GCS_BUCKET_NAME,
        "prefix": BUCKET_PREFIX,
        }, 
    "ttl": None,  ## only multiples of 604800 (1 week)
    "description": "This bucket was created with the UbiOps client library.",
    "labels": {},
}

response = api.buckets_create(PROJECT_NAME, bucket_data)
```

Our bucket is now ready, the last thing that we do before we start is create separate directories for each of the deployments.


```python
dir_name_1_reader = "dp-1-haystack-reader"
os.makedirs(dir_name_1_reader, exist_ok=True)

dir_name_2_converter = "dp-2-haystack-converter"
os.makedirs(dir_name_2_converter, exist_ok=True)

dir_name_3_summarizer = "dp-3-haystack-summarizer"
os.makedirs(dir_name_3_summarizer, exist_ok=True)

dir_name_4_writer = "dp-4-haystack-writer"
os.makedirs(dir_name_4_writer, exist_ok=True)
```

## 1. Creating the reader deployment

The first deployment that we will create will read all the files from the UbiOps bucket and create a list of local file paths that can be used to process the files in the next deployment.
- Input: `None`.
- Output: List of file URIs that map to files in the bucket.

The input of this deployment can be left empty because the deployment code reads from a bucket as determined by `BUCKET_NAME`, which is set as an environment variable. As long as a non-empty bucket with this name is created, the deployment can run successfully.

First, we will create the `deployment.py` script and `requirements.txt` file and zip them. After that we will create the deployment and upload the zip file to the deployment.


```python
INSTANCE_TYPE_1 = "256 MB + 0.0625 vCPU" 
DEPLOYMENT_NAME_1 = "1-haystack-reader"
ENVIRONMENT_NAME_1 = "1-haystack-reader"
DEPLOYMENT_VERSION_1 = "v1"
```

All the Python dependencies that you want to install in the deployment with `pip` need to be provided to the deployment in a `requirements.txt` file. In this first deployment only the UbiOps client library is required. The `os` library is also used in the deployment script, but this is a standard library that is provided by Python, so it does not have to be provided in the `requirements.txt`.


```python
%%writefile {dir_name_1_reader}/requirements.txt
ubiops
```

The deployment script below contains a `Deployment` class with two key methods:

- **`__init__` Method**  
  This method runs when the deployment starts. Here variables that we need during requests are defined, such as the name of the project. Furthermore, the UbiOps client is initialized inside the deployment and the file formats that are allowed are also defined here. For this tutorial we will process PDF files, but other file types are also supported by Docling, for example:
  - PDF
  - CSV
  - DOCX, XLSX, PPTX
  - Markdown	
  - AsciiDoc
  - HTML, XHTML

  Additionally, some image formats are also supported, as described in the [Docling documentation](https://docling-project.github.io/docling/usage/supported_formats/).

- **`request()` Method**  
  The request method contains the logic for processing incoming data. For our bucket with the name `BUCKET_NAME`, all files are listed, and those that adhere to the determined allowed file formats will be added to a list in the form of local file URIs. These are paths that map to documents in UbiOps storage buckets. In our case the file URI will be `ubiops-file://haystack-gcs/basic-text.pdf`, or a file name in this location of a file that you provided yourself.


```python
%%writefile {dir_name_1_reader}/deployment.py
import os
import ubiops

class Deployment:
    def __init__(self, base_directory, context):
        print("Initialising Deployment 1")
        self.PROJECT_NAME = context["project"]
        self.BUCKET_NAME = os.environ["BUCKET_NAME"]
        API_TOKEN = os.environ["API_TOKEN"]

        configuration = ubiops.Configuration()
        configuration.api_key['Authorization'] = API_TOKEN
        configuration.host = "https://api.ubiops.com/v2.1"
        client = ubiops.ApiClient(configuration)
        self.api = ubiops.CoreApi(client)

        self.allowed_file_formats = {'.pdf'} ## Other files types are also supported, check the Docling documentation for more information
        
    def request(self, data):
        print(f"Processing files from bucket '{self.BUCKET_NAME}'")
        response = self.api.files_list(
            project_name = self.PROJECT_NAME,
            bucket_name = self.BUCKET_NAME, ## this will include the prefix if that is set
            limit = 1000, ## the limit is max 1000
        )
           
        uri_list = []
        for file in response.files:
            _, ext = os.path.splitext(file.file)
            if ext in self.allowed_file_formats:
                uri = f'ubiops-file://{self.BUCKET_NAME}/{file.file}' ## already includes the prefix if set
                uri_list.append({"uri": uri})
        
        return uri_list
```

We zip the deployment script and the requirements file so that we can upload it to the platform later on.


```python
deployment_zip_path = shutil.make_archive(dir_name_1_reader, 'zip', dir_name_1_reader)
```

Now we can create the deployment for the reader. The input can be left empty, since the `BUCKET_NAME` is added as an environment variable. The output will be a list of URIs with `'data_type':'string'`. Labels and a description can be added.


```python
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME_1,
    description="",
    input_type="plain",
    output_type="structured",
    input_fields=[],
    output_fields=[{'name':'uri','data_type':'string'}],
    labels={},
)

deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)
```

After creating the deployment we create a version of the deployment. This is where the deployment package will be uploaded to. The instance type group is defined with the deployment name and can be set to the other available instance type groups, which you can find in the WebApp under Project settings > Intance type groups. 

Since we will not use the deployment continuously and directly (just in the pipeline), we can set `minimum_instances=0`, which means that the deployment will not start up directly and run continuously. Additionally, we can also set `maximum_idle_time=10`, meaning the deployment will only remain active for 10 seconds after processing a request. By setting `request_retention_mode="Full"`, we make sure that the inputs and outputs of the requests are stored.


```python
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION_1,
    language='python3-12',
    instance_type_group_name=INSTANCE_TYPE_1,
    minimum_instances=0,
    maximum_idle_time=10,
    request_retention_mode="Full",
)

version = api.deployment_versions_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME_1,
        data=version_template
)
```

The deployment requires certain variables to run which we do not have to hard-code into the `deployment.py` script. We have defined these variables earlier on in this tutorial, and now we can add them to the deployment as environment variables.
- The `API_TOKEN` is needed to establish a connection to the UbiOps client from within the deployment. We need to do this to make a call to the API to fetch the list of files in our bucket.
- The `BUCKET_NAME` is needed to fetch the files from the right location.


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_1,
    version=DEPLOYMENT_VERSION_1,
    data=ubiops.EnvironmentVariableCreate(name="API_TOKEN", value=API_TOKEN, secret=True),
)
```


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_1,
    version=DEPLOYMENT_VERSION_1,
    data=ubiops.EnvironmentVariableCreate(name="BUCKET_NAME", value=BUCKET_NAME, secret=False),
)
```

Now we can upload our zipped `deployment.py` script and `requirements.txt` file to the deployment version and wait for it to build.


```python
upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_1,
    version=DEPLOYMENT_VERSION_1,
    file=deployment_zip_path,
)
print(upload_response)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_1,
    version=DEPLOYMENT_VERSION_1,
    revision_id=upload_response.revision,
)
```

## 2. Creating the Docling converter deployment

The second step of this pipeline is converting the documents into JSON, providing the LLM that will summarize the documents with machine-readable content. This deployment employs [Docling](https://www.docling.ai/), a file conversion framework that turns documents of various file formats into well-structured text. It is designed specifically for LLM applications and produces readouts that preserve the layout information of a document. Because of its applications for LLMs, a Docling integration for Haystack is available, allowing for seamless integration of Docling into a Haystack pipeline. More information on `docling-haystack` can be found [here](https://haystack.deepset.ai/integrations/docling).

The converter deployment receives a list of file URIs from the first deployment, that maps to files in a UbiOps bucket. However, Docling cannot process files from URIs and requires local paths, therefore the files are downloaded directly into the base directory of the deployment before being converted. The JSON text that is produced by Docling is passed down to the next deployment, for summarization, and the file URIs used to locate the files in storage are passed on to the last deployment, to write the summary to the original files and store them in the bucket.

In the pipeline, prior to this deployment, a [subrequest operator](https://ubiops.com/docs/pipelines/operators/#create-subrequests) ensures that the list of files that was constructed in the first deployment is processed concurrently in each following deployment. All files will thus be processed by a deployment before being forwarded to the next deployment. Therefore the input of this deployment contains single files, while the output of the previous deployment contains a list of file URIs.
- Input: file URI that maps to a file in a bucket
- Output 1: JSON text
- Output 2: file URI that maps to a file in a bucket


```python
INSTANCE_TYPE_2 = "4096 MB + 1 vCPU" ## Docling is compute intensive, so requires a large instance
DEPLOYMENT_NAME_2 = "2-haystack-converter"
ENVIRONMENT_NAME_2 = "2-haystack-converter"
DEPLOYMENT_VERSION_2 = "v1"
```

Besides `ubiops`, we also need the `docling-haystack` and `haystack-ai` packages in our environment. Additionally, we need a minimum version of `sentence-transformers` for `haystack-ai`.


```python
%%writefile {dir_name_2_converter}/requirements.txt
docling-haystack
ubiops
haystack-ai
sentence-transformers>=5.0.0
```

When the deployment requires dependencies to be installed at the system level, they need to be provided in YAML file, to be installed by `sudo apt`. We will also zip this file together with the `requirements.txt` and the `deployment.py` script.


```python
%%writefile {dir_name_2_converter}/ubiops.yaml
apt:
  packages:
  - ffmpeg
```

Our deployment script is again constructed from an `__init__()` and a `request()` function. 

The `__init__()` function features the UbiOps client and the Docling converter being initialized. The UbiOps client is used when downloading the files in the request method with `ubiops.utils.download_file()`. Additionally, the `PROJECT_NAME` is defined, which is also needed in the `download_file()` function.

The `request()` method receives the file URIs as input and creates local output paths for the files to be downloaded to in the deployment as `/home/deployment/...`. These local paths are then fed into the Docling converter, which output text. Besides the JSON text, the file URIs are passed down as output to the last deployment in the same way that they came in, as `data['uri']`.

The output object that we obtain from using `DoclingConverter()` is a Haystack Document object, which analyzes the document in chunks. Furthermore, an embedder model from `haystack` is employed. This `SentenceTransformersTextEmbedder` model is warmed up during initialization. 

First we add the JSON text of every chunk together into `json_text`. Then we select the metadata of all of the chunks with `metadata = [doc.meta for doc in document_chunks]`. Afterwards, we transform the document's JSON text into embeddings. We can pass on the JSON content to the next component of the pipeline. The metadata and the embeddings were included to illustrate the possibilities of `haystack` and `docling-haystack`, they are outputted by this deployment and can be used as preferred, but they are not processed by the rest of the pipeline


```python
%%writefile {dir_name_2_converter}/deployment.py
import os
import ubiops
from docling_haystack.converter import DoclingConverter
from haystack.components.embedders import SentenceTransformersTextEmbedder

class Deployment:
    def __init__(self, base_directory, context):
        print("Initialising Deployment 2")
        self.PROJECT_NAME = context["project"]
        
        API_TOKEN = os.environ["API_TOKEN"]
        configuration = ubiops.Configuration()
        configuration.api_key['Authorization'] = API_TOKEN
        configuration.host = "https://api.ubiops.com/v2.1"
        self.client = ubiops.ApiClient(configuration)

        self.converter = DoclingConverter()

        self.text_embedder = SentenceTransformersTextEmbedder()
        self.text_embedder.warm_up()

    def request(self, data, context):
        print("Now processing", data['uri'])
        
        file_name = os.path.basename(data['uri'])
        output_path = f"/home/deployment/{file_name}"

        ubiops.utils.download_file(
            self.client,
            project_name=self.PROJECT_NAME,
            file_uri=data['uri'],
            output_path=output_path,
            stream=True,
            chunk_size=8192,
        )
        print("Downloaded file", output_path)
        
        document_chunks = self.converter.run(paths=[output_path])["documents"] ## the document will be analyzed in chunks

        json_text = "\n\n".join([doc.content for doc in document_chunks])
        metadata = [doc.meta for doc in document_chunks]
        embedded_text = self.text_embedder.run(json_text)["embedding"] ## beware that this model has a maximum sequence length of 512 tokens

        return {"json": {"text": json_text, "metadata": metadata, "embeddings": embedded_text}, "uri": data['uri']}
```

The `deployment.py` script, and the dependencies in the `.txt` and `.yaml` files, must be zipped before they can be uploaded to deployment.


```python
deployment_zip_path = shutil.make_archive(dir_name_2_converter, 'zip', dir_name_2_converter)
```

Now we can create the deployment on UbiOps. 

All outputs are in `"data_type":"string"`. Note that the `metadata` and the `embeddings` are not used in this pipeline, but were added to illustrate that they can produced at this stage by `haystack`.


```python
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME_2,
    description="",
    input_type="structured",
    output_type="structured",
    input_fields=[{'name':"uri","data_type":"string"}],
    output_fields=[{"name":"json","data_type":"string"},
                   {"name":"uri","data_type":"string"}]
    labels={},
)

deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)
```

Again, we need to create a version of this deployment.


```python
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION_2,
    language='python3-12',
    instance_type_group_name=INSTANCE_TYPE_2,
    minimum_instances=0,
    maximum_instances=2,
    maximum_idle_time=10,
    request_retention_mode="Full",
)

version = api.deployment_versions_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME_2,
        data=version_template
)
```

Since this deployment initializes the UbiOps client we need to provide an `API_TOKEN` as an environment variable to ensure authentication.


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_2,
    version=DEPLOYMENT_VERSION_2,
    data=ubiops.EnvironmentVariableCreate(name="API_TOKEN", value=API_TOKEN, secret=True),
)
```

Now we can upload our `deployment_package.zip` to our deployment. Since there are some more dependencies needed for this deployment it will require a bit longer to build the environment. This may take a few minutes before we can continue.


```python
upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_2,
    version=DEPLOYMENT_VERSION_2,
    file=deployment_zip_path,
)
print(upload_response)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_2,
    version=DEPLOYMENT_VERSION_2,
    revision_id=upload_response.revision,
    stream_logs=True,
)
```

## 3. Creating the summarizer deployment

The third deployment of the pipeline contains the another Haystack element, providing an OpenAI chat generator that can summarize the incoming texts. This summary is then passed on and added into the metadata of the documents to serve as annotation. 

[Haystack](https://docs.haystack.deepset.ai/docs/intro) is an AI-orchestration framework with many applications for LLM use-cases. The relatively small [`GPT-4 Omni Mini`](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) LLM is provided using the OpenAI agent of Haystack.
This deployment receives text as a string, generated by the Docling converter in the previous deployment, and feeds it into the LLM with the task:

    "Analyze the following document and produce annotations." 

The summary that is produced by the model is also a string and passed on to the next deployment as such.
- Input: document content in text
- Output: document summary in text


```python
INSTANCE_TYPE_3 = "4096 MB + 1 vCPU"
DEPLOYMENT_NAME_3 = "3-haystack-summarizer"
ENVIRONMENT_NAME_3 = "3-haystack-summarizer"
DEPLOYMENT_VERSION_3 = "v1"
```

For this deployment there are no system-level dependencies that need to be installed, only one Python-level dependency: `haystack-ai`.


```python
%%writefile {dir_name_3_summarizer}/requirements.txt
haystack-ai
```

This deployment script contains the following:
- **__init__()**:
The Haystack agent is initialized using `OpenAIChatGenerator` with model `gpt-4o-mini`. A system prompt for the model is also defined. The system prompt can be changed here if something else is preferred. The specific task that the model receives with each request can be changed in the `request()` function.

- **request()**:
The user prompt for the model is defined here. It contains a task, then the input data, and after that an output format for the summary. The response message is generated and selected to pass on to the next deployment.



```python
%%writefile {dir_name_3_summarizer}/deployment.py
import os
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool

class Deployment:
    def __init__(self, base_directory, context):
        print("Initialising Deployment 3")
        self.generator = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt="""
            You are an expert technical summarizer.
            Summarize the following documents into a coherent overview. 
            Focus on the main ideas and omit unnecessary details.
            """)

    def request(self, data, context):
        user_prompt = f"""
        TASK:
        Analyze the following document and produce annotations.

        INPUT DOCUMENT:
        \"\"\"
        {data['json']['text']}
        \"\"\"

        OUTPUT FORMAT (JSON):
        {{
        "summary": string,
        "key_topics": string[],
        "entities": {{
            "organizations": string[],
            "people": string[]
        }}}}
        """

        response = self.generator.run(messages=[ChatMessage.from_user(user_prompt)])
        
        return {"summary": next(msg for msg in response["messages"] if msg.role == "assistant").text}
```

We zip the deployment package.


```python
deployment_zip_path = shutil.make_archive(dir_name_3_summarizer, 'zip', dir_name_3_summarizer)
```

Next we create the deployment. Again the data types are strings.


```python
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME_3,
    description="",
    input_type="structured",
    output_type="structured",
    input_fields=[{"name":"json","data_type":"string"}],
    output_fields=[{"name":"summary","data_type":"string"}],
    labels={},
)

deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)
```

Next we create a version for this deployment with the selected instance type group.


```python
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION_3,
    language='python3-12',
    instance_type_group_name=INSTANCE_TYPE_3,
    minimum_instances=0,
    maximum_instances=2,
    maximum_idle_time=10,
    request_retention_mode="Full",
)

version = api.deployment_versions_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME_3,
        data=version_template
)
```

The `OPENAI_API_KEY` needs to be uploaded to the deployment environment for the deployment to run properly. How this key can be obtained is described at the start of this tutorial.


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_3,
    version=DEPLOYMENT_VERSION_3,
    data=ubiops.EnvironmentVariableCreate(name="OPENAI_API_KEY", value=OPENAI_TOKEN, secret=True),
)
```

Upload the deployment package and wait for the deployment to build before continuing.


```python
upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_3,
    version=DEPLOYMENT_VERSION_3,
    file=deployment_zip_path,
)
print(upload_response)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_3,
    version=DEPLOYMENT_VERSION_3,
    revision_id=upload_response.revision,
    stream_logs=True,
)
```

## 4. Creating the writer deployment

The last deployment of the pipeline is the writer deployment, which will annotate the files by adding the summary as metadata. To do this we need to read the file from the file URI that we used earlier, fetch the summary and add it to the metadata, and then save the file. We save the modified file in pur synced UbiOps bucket, but separate from the original in a folder called `processed_files`.

The input for this deployment comes from the second deployment, where the documents were converted into JSON, which passes a file URI, and from the previous deployment, which passes the summary that was made. The output will be the paths of the modified files.
- Input 1: file URI that maps to the original file in the bucket
- Input 2: summary text
- Output: file path that maps to the modified file in the bucket


```python
INSTANCE_TYPE_4 = "512 MB + 0.125 vCPU"  # You can find all possible Instance type groups in the WebApp under Project Admin > Project settings > Instance type groups
DEPLOYMENT_NAME_4 = "4-haystack-writer"
ENVIRONMENT_NAME_4 = "4-haystack-writer"
DEPLOYMENT_VERSION_4 = "v1"
```

This deployment only requires the `pypdf` Python package, to read the PDF documents and write new metadata in them.


```python
%%writefile {dir_name_4_writer}/requirements.txt
pypdf
```

This deployment script features two actions during the initialization, in `__init__()`, the establishment of the `BUCKET_NAME` and the initialization of the `PdfWriter()` class from `pypdf`. 

The `request()` method features reading the pages of the PDF from the file URI, writing the summary into the metadata, and the storing the file separately. The files are added to the `processed_files` folder have `-incl-metadata` added to the file name. The structure of the returned object ensures that the file is stored properly. More information on this can be found in the [documentation](https://ubiops.com/docs/input-output/files-deployment/#handling-file-output-in-the-deployment-code).


```python
%%writefile {dir_name_4_writer}/deployment.py
import os
from pypdf import PdfReader, PdfWriter
from pathlib import Path

class Deployment:
    def __init__(self, base_directory, context):
        print("Initialising Deployment 4")
        self.BUCKET_NAME = os.environ['BUCKET_NAME']
        self.writer = PdfWriter()

    def request(self, data, context):
        summary = data['summary']
        file_path = data['uri']

        reader = PdfReader(file_path)
        for page in reader.pages:
            self.writer.add_page(page)
        
        metadata = dict(reader.metadata or {})
        metadata["/MyApp:Summary"] = summary

        self.writer.add_metadata(metadata)
        
        path = Path(file_path)

        with open(file_path, "wb") as file:
            self.writer.write(file)

        return {
            "output_file": {
                "file": file_path,
                "bucket": self.BUCKET_NAME,
                "bucket_file": f"processed_files/{path.stem}-incl-metadata{path.suffix}"
            }
        }
```


```python
deployment_zip_path = shutil.make_archive(dir_name_4_writer, 'zip', dir_name_4_writer)
```

The last deployment can now be created. For the input of the summary, we use `"data_type":"string"`, the same data type as for the output of the summary in the last deployment. However, the file URI, which has `"data_type":"string"` at the output of the converter deployment, has `"data_type":"file"` at the input of this deployment. By doing this we ensure that the file URI is passed on and read as a string, but that the file that it maps to becomes available in the current deployment.

The output data type is also set to file, so that the modified file will be uploaded into the bucket at the end of the pipeline.


```python
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME_4,
    description="",
    input_type="structured",
    output_type="structured",
    input_fields=[{"name":"uri","data_type":"file"},
                  {"name":"summary","data_type":"string"}],
    output_fields=[{"name":"output_file","data_type":"file"}],
    labels={},
)

deployment = api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)
```

We again need to create a version of the deployment that we just made.


```python
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION_4,
    language='python3-12',
    instance_type_group_name=INSTANCE_TYPE_4,
    minimum_instances=0,
    maximum_instances=2,
    maximum_idle_time=10,
    request_retention_mode="Full",
)

version = api.deployment_versions_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME_4,
        data=version_template
)
```

The only environment variable that is needed in this deployment is the `BUCKET_NAME`, so that the files are written to the same bucket that they are read from.


```python
api_response = api.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_4,
    version=DEPLOYMENT_VERSION_4,
    data=ubiops.EnvironmentVariableCreate(name="BUCKET_NAME", value=BUCKET_NAME, secret=False),
)
```

And we upload the deployment package to the deployment and wait for it to build.


```python
upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_4,
    version=DEPLOYMENT_VERSION_4,
    file=deployment_zip_path,
)
print(upload_response)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME_4,
    version=DEPLOYMENT_VERSION_4,
    revision_id=upload_response.revision,
    stream_logs=True,
)
```

## 5. Granting bucket permissions

All the deployments have now been created and are now visible on the [WebApp](https://app.ubiops.com/sign-in). However, before we can send a requests to the deployments we need to give them permission to read and write from our synced bucket. Specifically, the reader deployment needs permission to read from the bucket and the writer deployment permission to read and write from and into the bucket. The two haystack deployments in between process do not require any permissions.

Permissions are granted by making a request to the API and using the `role_assignments_create()` function. We specify the assignee and the resource and the types of both. By assigning the role `files-reader` we grant permission to read.


```python
reader_permission = {
    "assignee": DEPLOYMENT_NAME_1,
    "assignee_type": "deployment",
    "role": "files-reader",
    "resource": BUCKET_NAME,
    "resource_type": "bucket"
}

response_reader = api.role_assignments_create(
    project_name=PROJECT_NAME,
    data=reader_permission
)
print(response_reader)
```

The setup for the writer deployment is the same, but we assign the role `files-writer`, to grant both reading and writing permissions.


```python
writer_permission = {
    "assignee": DEPLOYMENT_NAME_4,
    "assignee_type": "deployment",
    "role": "files-writer",
    "resource": BUCKET_NAME,
    "resource_type": "bucket"    
}

response_writer = api.role_assignments_create(
    project_name=PROJECT_NAME,
    data=writer_permission
)
print(response_writer)
```

## 6. Creating the pipeline and sending a request

Now that the deployments have been added to UbiOps and have been granted the required permissions, they are operational. The only thing left to do is connecting the deployments into a pipeline and sending a request. 

The `PipelineCreate()` class is similar the `DeploymentCreate()` class that we used earlier on. We specify a name, input and output fields, including data types, and a description or labels if we want. The input and output fields must match with the first and last deployment of our pipeline to connect them to the start and end of the pipelines.
- The `input_fields` of the first deployment were left empty, so we select `input_type="plain"` and keep the input fields empty as well.
- The `output_fields` of the last deployment were set to `"data_type":"file"`, so we make sure the output type and fields of the pipeline match this.


```python
PIPELINE_NAME = 'haystack-annotation'
PIPELINE_VERSION = "v1"

pipeline_template = ubiops.PipelineCreate(
    name=PIPELINE_NAME,
    description="",
    input_type="plain",
    input_fields=[],
    output_type="structured",
    output_fields=[{"name":"output_file","data_type":"file"}],
    labels={},
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template,
)
```

Pipelines can either be constructed by hand in the WebApp or through a call to the API. We need to pass the structure of the pipeline as data when calling the API to make the pipeline. 

Below you can see the structure of our pipeline, first listing the separate objects and then describing each of the attachments (how each object maps to another). The `request_retention_mode` is set to full to store all inputs and outputs and the `request_retention_time` is set to 604800 seconds, which is 1 week.

Besides the deployments a [subrequest operator](https://ubiops.com/docs/pipelines/operators/#create-subrequests) is also added into the pipeline, placed between the first and second deployment. The first deployment fetches all the files that will be processed and passes them on as a list. From that point on we want to process the whole list of files in parallel in each deployment, instead of running them individually one-by-one. We can achieve by setting two variables: `batch_size`, determined at the subrequest operator, and `maximum_instances`, determined at the deployments.

The batch size determines the maximum size of the batches of the requests, and the maximum number of instances determines how many instances will process the batches. For example, when you have 12 files in total and set `batch_size` to 4, the list of 12 files will be split into 3 batches of 4 files. You can now spin up three instances of your deployment to process the 3 lists you created by setting `maximum_instances` to 3. 

When you set `batch_size` to `None`, the subrequest operator will parallelize the list by maximizing the number of batches, thus splitting the list of requests into the largest number of subrequests.

In this tutorial we process 4 files and we have set `maximum_instances=2`. So when we set `batch_size` to `None`, the 4 requests will be partitioned in batches of 2 over both instances.


```python
data = {
    "pipeline_name": PIPELINE_NAME,
    "version_name": PIPELINE_VERSION,
    "request_retention_mode": "full",
    "request_retention_time": 604800,
    "objects": [
        {
            "name": DEPLOYMENT_NAME_1,
            "reference_name": DEPLOYMENT_NAME_1,
            "reference_type": "deployment",
            "reference_version": DEPLOYMENT_VERSION_1,
        },
        {
            "name": DEPLOYMENT_NAME_2,
            "reference_name": DEPLOYMENT_NAME_2,
            "reference_type": "deployment",
            "reference_version": DEPLOYMENT_VERSION_2,
        },
        {
            "name": DEPLOYMENT_NAME_3,
            "reference_name": DEPLOYMENT_NAME_3,
            "reference_type": "deployment",
            "reference_version": DEPLOYMENT_VERSION_3,
        },
        {
            "name": DEPLOYMENT_NAME_4,
            "reference_name": DEPLOYMENT_NAME_4,
            "reference_type": "deployment",
            "reference_version": DEPLOYMENT_VERSION_4,
        },
        {
            "name": "operator-one-to-many",
            "reference_name": "one-to-many",
            "reference_type": "operator",
            "configuration": {
                "batch_size": None,  ## This is where you can set the batch size
                "input_fields": [{"name": "uri", "data_type": "string"}],
                "output_fields": [{"name": "uri", "data_type": "string"}]
            }
        }
    ],
    "attachments": [
        ## connect start to reader (1)
        {
            "destination_name": DEPLOYMENT_NAME_1,
            "sources": [{
                "source_name": "pipeline_start",
                "mapping": []
            }]
        },
        ## connect reader (1) to subrequest operator
        {
            "destination_name": "operator-one-to-many",
            "sources": [{
                "source_name": DEPLOYMENT_NAME_1,
                "mapping": [{
                    "source_field_name": "uri",
                    "destination_field_name": "uri",
                }]
            }]
        },
        ## connect subrequest operator to converter (2)
        {
            "destination_name": DEPLOYMENT_NAME_2,
            "sources": [{
                "source_name": "operator-one-to-many",
                "mapping": [{
                    "source_field_name": "uri",
                    "destination_field_name": "uri",
                }]
            }]
        },
        ## connect converter (2) to summarizer (3)
        {
            "destination_name": DEPLOYMENT_NAME_3,
            "sources": [{
                "source_name": DEPLOYMENT_NAME_2,
                "mapping": [{
                    "source_field_name": "json",
                    "destination_field_name": "json",                    
                }]
            }]
        },
        ## connect summarizer (3) and converter (2) to writer (4)
        {
            "destination_name": DEPLOYMENT_NAME_4,
            "sources": [{
                "source_name": DEPLOYMENT_NAME_3,
                "mapping": [{
                    "source_field_name": "summary",
                    "destination_field_name": "summary",                    
                }]
            },
            {
                "source_name": DEPLOYMENT_NAME_2,
                "mapping": [{
                    "source_field_name": "uri",
                    "destination_field_name": "uri",    
                }]
            }]
        },
        ## connect writer (4) to end
        {
            "destination_name": "pipeline_end",
            "sources": [{
                "source_name": DEPLOYMENT_NAME_4,
                "mapping": [{
                    "source_field_name": "output_file",
                    "destination_field_name": "output_file",    
                }]
            }]            
        }
    ]
}
```

We can now upload the pipeline structure to UbiOps.


```python
pipeline_template = ubiops.PipelineVersionCreate(
    version=PIPELINE_VERSION,
    request_retention_mode="full",
    objects=data['objects'],
    attachments=data['attachments'],
)

api.pipeline_versions_create(
    project_name=PROJECT_NAME, pipeline_name=PIPELINE_NAME, data=pipeline_template
)
```

Everything is now ready to send a request to the pipeline and process the files in our bucket. Make sure you properly executed every cell in this tutorial and that you have added a file to process to your GCS/UbiOps bucket.

The input of the request can be left empty, as the `BUCKET_NAME` is provided to the first deployment as an environment variable.


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=PIPELINE_NAME,
    data={}
)
```

We now sent a request to the pipeline. The results of this request will become visible in the [WebApp](https://app.ubiops.com/sign-in) under Pipelines. You can go the version of the pipeline that we created (`v1`) and click on the Requests tab to view the request that we just made. It may take a minute before completing. Check the logs of the pipeline to see what step it is running at the moment. Once the request is finished, and completed the processed file(s) will be visible in your GCS bucket and in the UbiOps bucket that it is synced with. You can find the UbiOps bucket in the WebApp in the Storage tab.

## Summary

In this tutorial, you learned how to:

1. **Couple GCS bucket to UbiOps** to directly access files from an outside bucket
2. **Employ Haystack pipeline elements in UbiOps deployments** for Docling PDF conversions and LLM summaries. 
3. **Construct a pipeline** from individual deployments using the UbiOps library for document enrichment.

## Resources

- [UbiOps Pipeline Documentation](https://ubiops.com/docs/pipelines/)
- [UbiOps Pipeline Tutorial](https://ubiops.com/docs/ubiops_tutorials/pipeline-tutorial/pipeline-tutorial/)
- [Haystack documentation](https://docs.haystack.deepset.ai/docs/intro)
- [Docling-Haystack](https://haystack.deepset.ai/integrations/docling)

