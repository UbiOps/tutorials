# Fine-tuning Falcon 1B 

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/finetuning-falcon/finetuning-falcon/finetuning-falcon.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/finetuning-falcon/finetuning-falcon/finetuning-falcon.ipynb){ .md-button .md-button--secondary }


This notebook shows how you can fine-tune the Falcon 1B model from Huggingface on 
[English quotes](https://huggingface.co/datasets/Abirate/english_quotes) using the UbiOps 
[training functionallity](https://ubiops.com/docs/training/). In order to fine-tune the model we'll need to create an 
experiment which defines the training set up, in this experiment we willll then iniate training runs which are the 
actual code executions. In the case of this notebook the code inside these training runs will be executed on the dataset 
specified earlier. For this guide you will also need to have initialized the training functionallity inside your project, 
which can be done by going to the **Training** page and clicking **Enable training**. 

Note that the aim of this guide is to show you ***how*** you can fine-tune Falcon, as such Falcon will not be fine-tuned 
for a specific benchmark.

The fine-tuning will be done in four steps:
1. Connecting with the UbiOps API client
2. Create the environment for training experiment
3. Create the training experiment
4. Initialize two training runs (more explanation will follow below)

## 1) Connecting with the UbiOps API client

To use the UbiOps API from our notebook, we need to install the UbiOps Python client library.


```python
!pip install --upgrade ubiops
```

To set up a connection with the UbiOps platform API we need the name of your UbiOps project and an API token with `project-editor` permissions.

Once you have your project name and API token, paste them below in the following cell before running.


```python
import ubiops

API_TOKEN = '<API TOKEN>' # Make sure this is in the format "Token token-code"
PROJECT_NAME = '<PROJECT_NAME>'    # Fill in your project name here

configuration = ubiops.Configuration()
configuration.api_key["Authorization"] = API_TOKEN

api_client = ubiops.ApiClient(configuration)
api = ubiops.api.CoreApi(api_client)
```

## 2. Create the environment for training experiment

An environment on UbiOps is built up out of a `base environment`, which for this tutorial will be `ubuntu22-04-python3-11-cuda11-7-1` 
to which we can add aditional dependencies. The packages
that will be used inside the deployment will be defined in a `requirements.txt`. For this guide a package from `git` is also
required, [which can be done](https://ubiops.com/docs/howto/howto-load-from-git/) by creating a `ubiops.yaml`, the `ubiops.yaml` can be used for packages that need to be downloaded
on OS-level. These files will be added to a directory, zipped, and then uploaded to UbiOps.


```python
!mkdir fine-tuning-environment-files
```

### 2a) Create the `requirements.txt`


```python
%%writefile fine-tuning-environment-files/requirements.txt
ubiops
joblib
torch
scipy                      
bitsandbytes
git+https://github.com/huggingface/transformers.git
git+https://github.com/huggingface/peft.git
git+https://github.com/huggingface/accelerate.git

```

### 2b) Create the `ubiops.yaml`


```python
%%writefile fine-tuning-environment-files/ubiops.yaml
apt:
  packages:
    - git
```

Now we need to zip the fine-tuning-environment-files directory, and define the coding environment so we can upload it to
UbiOps.


```python
import shutil

zip_name = "fine-tuning-environment-files"
ENVIRONMENT_NAME = "fine-tuning-falcon1b"
shutil.make_archive(zip_name, "zip", "fine-tuning-environment-files")
```


```python
# Define the environment and upload the zip file to UbiOps
data = ubiops.EnvironmentCreate(
    name=ENVIRONMENT_NAME, base_environment="ubuntu22-04-python3-11-cuda11-7-1"
)

api_response = api.environments_create(PROJECT_NAME, data)
print(api_response)
api_response = api.environment_revisions_file_upload(
    PROJECT_NAME, ENVIRONMENT_NAME, file=f"{zip_name}.zip"
)
print(api_response)

# Wait for the environment to finish building
ubiops.utils.wait_for_environment(
    client=api_client,
    project_name=PROJECT_NAME,
    environment_name=ENVIRONMENT_NAME,
    timeout=1800,
    stream_logs=True,
)
```

## 3) Create the training experiment & environment variables

Now that the environment has been created we can start defining the training set-up, i.e., the `experiment`. Here we define
the environment, the instance type and the storage location (bucket) for the training runs. Defining the instance type and environment 
on experiment level makes it possible to apply techniques like [hyper-parameter tuning](https://ubiops.com/docs/ubiops_tutorials/xgboost-training/xgboost-training/).
If you do not have access to GPU, you will need to change the instance type below to `16384mb_t4`


```python
from datetime import datetime
from ubiops.training.training import Training


training_instance = Training(api_client)

# Create experiment
EXPERIMENT_NAME = f"falcon-fine-tuning-{datetime.now().date()}"

api_response = training_instance.experiments_create(
    project_name=PROJECT_NAME,
    data=ubiops.ExperimentCreate(
        name=EXPERIMENT_NAME,
        instance_type_group_name="16384 MB + 4 vCPU + NVIDIA Tesla T4",  # You can use '16384 MB + 4 vCPU' if you run on CPU
        description="A finetuning experiment for Falcon",
        environment=ENVIRONMENT_NAME,
        default_bucket="default",
        labels={"type": "pytorch", "model": "flaconLLM"},
    ),
)

# Wait for the experiment to finish building
ubiops.utils.wait_for_experiment(
    client=api_client,
    project_name=PROJECT_NAME,
    experiment_name=EXPERIMENT_NAME,
    timeout=1800,
    stream_logs=True,
)
```

Now we create an environment variable so we can access the *default* bucket. The results of the model will be stored inside this bucket.


```python
# Create an environment variable for the api token
envvar_projname = ubiops.EnvironmentVariableCreate(
    name="API_TOKEN", value=API_TOKEN, secret=True
)
api.deployment_version_environment_variables_create(
    PROJECT_NAME,
    deployment_name="training-base-deployment",
    version=EXPERIMENT_NAME,
    data=envvar_projname,
)

# Create an environment variable for the project name
envvar_projname = ubiops.EnvironmentVariableCreate(
    name="PROJECT_NAME", value=PROJECT_NAME, secret=False
)
api.deployment_version_environment_variables_create(
    PROJECT_NAME,
    deployment_name="training-base-deployment",
    version=EXPERIMENT_NAME,
    data=envvar_projname,
)
```

## 4) Set up the training runs

The training runs are the actual code executions on a specific dataset. For each run you can configure different training 
code (in the form of a `train.py`), training data and different parameters.

For this experiment we will initiate two training runs, which are the actual code executions:
- A preparation run (`prepare.py`) in which will the models' checkpoints and dataset will be downloaded, after which 
they will be stored inside the (*default*) UbiOps bucket which we defined when we created the experiment.
- A training run (`train.py`) in which these files will be downloaded, and then used to fine-tune the Falcon 1B model. The
 model will be stored in the same bucket as the models' weights and the dataset from the preparation run.

#### 4a) Create and initialize the `prepare.py`

First we create the `prepare.py`:


```python
%%writefile prepare.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import tarfile
import ubiops
import requests
import os 

def train(training_data, parameters, context = {}):

    configuration = ubiops.Configuration(api_key={'Authorization': os.environ["API_TOKEN"]})
    api_client = ubiops.ApiClient(configuration)
    api = ubiops.api.CoreApi(api_client)
    
    # Load model weights
    print("Load model weights")
    model_id = "tiiuae/falcon-rw-1b"
    cache_dir = "checkpoint"
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = cache_dir)
    with tarfile.open(f'{cache_dir}.tar', 'w') as tar: 
         tar.add(f"./{cache_dir}/")
         
    # Uploading weights
    print('Uploading weights')
    file_uri = ubiops.utils.upload_file( 
      client=api_client, 
      project_name=os.environ["PROJECT_NAME"], 
      file_path=f'{cache_dir}.tar', 
      bucket_name="default", 
      file_name=f'{cache_dir}.tar'
    )

    # Load dataset
    print("Load dataset")
    ds = "quotes.jsonl"
    r = requests.get("https://huggingface.co/datasets/Abirate/english_quotes/resolve/main/quotes.jsonl")
    with open(ds, 'wb') as f:
        f.write(r.content)
        
    # Uploading dataset
    file_uri = ubiops.utils.upload_file( 
      client=api_client, 
      project_name=os.environ["PROJECT_NAME"], 
      file_path=ds, 
      bucket_name="default", 
      file_name=ds
    )
```

Then we initialize the a training run which will execute the code inside the `prepare.py`:


```python
run = training_instance.experiment_runs_create(
    project_name=PROJECT_NAME,
    experiment_name=EXPERIMENT_NAME,
    data=ubiops.ExperimentRunCreate(
        name="Load",
        description="Load model",
        training_code="./prepare.py",
        parameters={}
    ),
    timeout=14400
)

# Wait for the prepare.py run to complete
ubiops.utils.wait_for_experiment_run(
    client=api_client,
    project_name=PROJECT_NAME,
    experiment_name=EXPERIMENT_NAME,
    run_id=run.id,
)
```

#### 4b) Create and intialize the `train.py`

As with the `prepare.py` we first define the code for the `train.py`:


```python
%%writefile train.py

import ubiops
import os
import tarfile
import json
import joblib
from typing import List
import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from ubiops import utils

class QuotesDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.tokenizer(self.data[idx]["quote"])

def train(training_data, parameters, context = {}):

    configuration = ubiops.Configuration(api_key={'Authorization': os.environ["API_TOKEN"]})
    api_client = ubiops.ApiClient(configuration)
    api = ubiops.api.CoreApi(api_client)       
  # First step is to load model weights and and dataset of english quotes into deployment
    for f in ["checkpoint.tar","quotes.jsonl"]: 
        file_uri = ubiops.utils.download_file(
          client= api_client, #a UbiOps API client, 
          file_name=f,
          project_name=os.environ["PROJECT_NAME"],
          output_path=".",
          bucket_name="default"
        )

    with tarfile.open("checkpoint.tar", 'r') as tar:
        tar.extractall(path=".")
    
    # This config allow to represent model in a lower percision. It means every weight in it is going to take 4bits instead 32bit. So we will use ~ 8 times less vram.
    nf4_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_id="tiiuae/falcon-rw-1b"
    cache_dir = "checkpoint"

    # Loading model weights and allocating them according to config
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, quantization_config=nf4_config)
  
    # Also enabling checkpointing, a technique that allows us to save memory by recomputing some nodes multiple times.
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Lora is another technique that allows to save memory. However this time by reducing absolute number of trainable parameters. It also defines a task for our fine tuning as CAUSAL_LM, it means llm will learn to perdict next word based on previous words in a quote.
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    tokenizer.pad_token = tokenizer.eos_token

    lines = list()
    with open("quotes.jsonl", 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    
    dataset = QuotesDataset(lines, tokenizer)

    # Run trainer from the transformers library.
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # Silence the warnings. Please re-enable for inference!
    finetuned_model = trainer.train()

    # Save the model
    joblib.dump(finetuned_model, "finetuned_falcon.pkl")

    file_uri = ubiops.utils.upload_file( 
      client=api_client, 
      project_name=os.environ["PROJECT_NAME"], 
      file_path=f'{cache_dir}.tar', 
      bucket_name="default", 
      file_name=f'{cache_dir}.tar'
    )
    
    return {"artifact": "finetuned_falcon.pkl"}

```

Then we initialize a training run which will execute the code inside the `train.py`:


```python
response_train_run = training_instance.experiment_runs_create(
    project_name=PROJECT_NAME,
    experiment_name=EXPERIMENT_NAME,
    data=ubiops.ExperimentRunCreate(
        name="Train",
        description="training run",
        training_code="./train.py"
    ),
    timeout=14400
)
```

## And there you have it!

We have just fine-tuned the Falcon 1B model from Huggingface on the `quotes.json` dataset. The model can be accessed by 
going to the **Train** run inside the **falcon-fine-tuning** experiment we created and clicking on the link from the 
**Output artefact**. From there you can download the model by clicking on the **Download** button.

You can also copy the `file_uri` by clicking on the **copy** button. The `file_uri` can then be used to import the model 
inside a deployment, by using something like the code snippet below:
```
ubiops.utils.download_file(api_client,
                        PROJECT_NAME,
                        file_uri=file_uri,
                        output_path='checkpoint.tar'
                        )
```

More information about using files in a deployment can be found in [our documentation](https://ubiops.com/docs/input-output/). We also provide several Howto's that explain how you can use files inside a deployment, these can be found on the bottom of the Storage documentation page.
