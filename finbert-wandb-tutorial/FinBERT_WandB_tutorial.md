# Train and inference FinBERT on an MLOps pipeline by combining Weights & Biases and UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/blob/master/finbert-wandb-tutorial/finbert-wandb-tutorial/FinBERT_WandB_tutorial.ipynb){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/finbert-wandb-tutorial/finbert-wandb-tutorial/FinBERT_WandB_tutorial.ipynb){ .md-button .md-button--secondary }

In this notebook we will cover how to

1.   Use UbiOps for model training, hyperparameter tuning and running inference
2.   Use W&B for experiment tracking, model evaluation and comparison
3.   Use W&B as a model registry to track models that have been moved to inference on UbiOps
4.   Use UbiOps to transform the model in a live and scalable API

FinBERT, a pre-trained natural language processing (NLP) model, is designed to assess the sentiment of financial text. It is built by training the BERT language model in the finance domain on a substantial financial dataset. In this example, we will fine-tune it on a custom dataset. The model provides softmax outputs corresponding to three labels: positive, negative, or neutral sentiment. This model can be used through the library of HuggingFace `transformers`.

We are going to use the UbiOps platform to fine-tune FinBERT on different hyperparameter configurations on a financial news dataset, in parallel, in the cloud, using CPUs. FinBERT is a generic model, thus one would also want to further fine-tune it on their own datasets. We will do three training jobs to find the best combination of hyperparameters, thus finding the best training flow.

While the training jobs are running, we head over to Weights & Biases to analyze performance metrics during our training runs, and to compare the final models. After checking the accuracy metrics of all three training runs, we will store our best performing ML model on the Weights & Biases Model Registry, and deploy it by turning it into a live and scalable API endpoint on UbiOps. The model can be conveniently exposed to end-users via the API endpoint in a production set-up, allowing it to scale, depending on the demand.

If you have a paid subscription account, you may upgrade your CPU instance to a GPU instance (e.g. `instance_type_group_name = "16384 MB + 4 vCPU + NVIDIA Tesla T4"`)! You would also need to select an environment with CUDA compiled. More on this matter can be read [here](https://ubiops.com/docs/deployments/gpu-deployments/).

The FinBERT model can be found [here](https://huggingface.co/ProsusAI/finbert) and the dataset [here](https://huggingface.co/datasets/Jean-Baptiste/financial_news_sentiment).
The dataset we use is from the HuggingFace `datasets` library, and it consists of ~2000 Canadian news articles with manually validated financial sentiment. It also has a topic label, which can be used for further experiments!


```python
# This step may take a while
!pip install -qU wandb
!pip install -qU ubiops
```


```python
import ubiops


API_TOKEN = ""  # Make sure this is in the format "Token token-code"
PROJECT_NAME = ""  # You can name your UbiOps project however you want, but it must be globally unique and created in advance.

ENVIRONMENT_NAME = "finbert-environment"
EXPERIMENT_NAME = "finbert-training"

INFERENCING_DEPLOYMENT_NAME = "finbert-inference"
INFERENCING_DEPLOYMENT_VERSION = "v1"

WANDB_ENTITY = "" # this is either your W&B username, or a W&B team you are part of.
WANDB_PROJECT = "finbert-training"
WANDB_API_KEY = "" # You can get your API key here: https://wandb.ai/authorize
```

Set up a connection to the UbiOps API.


```python
configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key["Authorization"] = API_TOKEN

api_client = ubiops.ApiClient(configuration)
core_instance = ubiops.CoreApi(api_client=api_client)
training_instance = ubiops.Training(api_client=api_client)
print(core_instance.service_status())
```

Below we will build a training job that the API of UbiOps understands. A training job in UbiOps is called a run. To run the training job on UbiOps, we need to create a file named `train.py` and include our code here. This code will execute as a single *Run* as part of an *Experiment*. An *Experiment* can contain multiple training runs. Training runs inside the experiment run on top of an *Environment*. The *Environment* contains an instance type (hardware) and code dependencies. So let us start with making a directory store to store our environment instructions. Here, it contains a `requirements.txt` that contains the Python dependencies that our code needs to be able to run. In this case we use TensorFlow 2.13.0.


```python
!mkdir environment
```


```python
%%writefile environment/requirements.txt
datasets
tensorflow==2.13.0
transformers
wandb
```

Let's take a look at the training script. The script needs to contain a `train()` function, with input parameters `training_data` (a file path to your training data) and `parameters` (a dictionary that contains the parameters of your choice). The `training_data` path can be set to `None`, in case data is grabbed from an external location, such as an online object storage, or from a data science package, as we do in this example.


```python
!mkdir training_code
```


```python
%%writefile training_code/train.py
import os

import tensorflow as tf
import wandb

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TFBertForSequenceClassification,
    DataCollatorWithPadding,
)


def train(training_data, parameters, context):
    # Prepare the stock headlines datasets
    dataset = load_dataset("Jean-Baptiste/financial_news_sentiment")

    # Split them into train and test datasets
    train_ds = dataset["train"]
    test_ds = dataset["test"]

    hyperparameters = dict(
        epochs=parameters.get("nr_epochs", 10),
        batch_size=parameters.get("batch_size", 32),
        learning_rate=parameters.get("learning_rate", 2e-5),
        weight_decay=parameters.get("weight_decay", 0.01),
    )

    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_api_key = os.getenv("WANDB_API_KEY")

    wandb.login(key=wandb_api_key)
    with wandb.init(entity=wandb_entity, project=wandb_project, config=hyperparameters) as train_run:
        # Get FinBERT model using transformers
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        finbert = TFBertForSequenceClassification.from_pretrained("ProsusAI/finbert")

        print(f"This is the original finbert model with details: {finbert.summary()}")
        print(f"Type of finbert{type(finbert)}")

        # Tokenize all dataset without padding
        train_ds = train_ds.map(lambda x: tokenizer(x["title"]), batched=True)

        # Convert HuggingFace dataset to TF Data and combine sentences into batches with padding
        train_ds = train_ds.to_tf_dataset(
            columns=["input_ids", "token_type_ids", "attention_mask"],
            label_cols="labels",
            batch_size=train_run.config.batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer=tokenizer, return_tensors="tf"
            ),
            shuffle=True,
        )

        # Also convert the test dataset
        test_ds = test_ds.map(lambda x: tokenizer(x["title"]), batched=True)
        test_ds = test_ds.to_tf_dataset(
            columns=["input_ids", "token_type_ids", "attention_mask"],
            label_cols="labels",
            batch_size=train_run.config.batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer=tokenizer, return_tensors="tf"
            ),
            shuffle=False,
        )

        # Compile and fit model on the training dataset
        finbert.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=train_run.config.learning_rate,
                weight_decay=train_run.config.weight_decay,
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        finbert.fit(
            x=train_ds,
            epochs=train_run.config.epochs,
            callbacks=[wandb.keras.WandbCallback()],
            validation_data=test_ds,
        )

    # Evaluate the model
    final_loss, final_accuracy = finbert.evaluate(x=test_ds)

    # Save the model file with tf.keras
    finbert.save("finbert.keras")

    return {
        "artifact": "finbert.keras",
        "metadata": {},
        "metrics": {"accuracy": final_accuracy, "loss": final_loss},
        "additional_output_files": [],
    }
```

Now we zip our environment.

```python
import shutil

environment_archive = shutil.make_archive(
    "environment", "zip", "finbert-wandb-tutorial", "environment"
)
```

Let's enable training and create the environment! This needs to be done once in your project.



```python
try:
    training_instance.initialize(project_name=PROJECT_NAME)
except ubiops.exceptions.ApiException as e:
    print(
        f"The training feature may already have been initialized in your project: {e}"
    )
```

Let's create a new environment now!


```python
core_instance.environments_create(
    project_name=PROJECT_NAME,
    data=ubiops.EnvironmentCreate(
        name=ENVIRONMENT_NAME,
        display_name=ENVIRONMENT_NAME,
        base_environment="ubuntu22-04-python3-11",
        description="Environment with TensorFlow 2.13, wandb and HuggingFace libraries",
    ),
)
```

Finally, we upload our environment archive to UbiOps.


```python
core_instance.environment_revisions_file_upload(
    project_name=PROJECT_NAME,
    environment_name=ENVIRONMENT_NAME,
    file=environment_archive,
)
```

Note that building an environment can take long if this is the first time, because all packages from the
`requirements.txt` need to be installed inside the environment. This is a one-time process per environment.


```python
ubiops.utils.wait_for_environment(
    core_instance.api_client, PROJECT_NAME, ENVIRONMENT_NAME, timeout=1800, stream_logs=True
)
```

Let's create an experiment. Experiments segment different training runs. We select our compute resources to have 8GB of RAM. When we upload a training job, the training code will be run on top of our environment on the selected compute resource. Within this experiment, we can easily try out different training codes, or run the same training code with different hyperparameters. In this example, we will do the latter.


```python
experiment = training_instance.experiments_create(
    project_name=PROJECT_NAME,
    data=ubiops.ExperimentCreate(
        instance_type_group_name="8192 MB + 2 vCPU",  # Change this to "16384 MB + 4 vCPU + NVIDIA Tesla T4" if you want to use GPUs
        description="FinBERT training experiment runs",
        name=EXPERIMENT_NAME,
        environment=ENVIRONMENT_NAME,
        default_bucket="default",

    ),
)
ubiops.utils.wait_for_experiment(core_instance.api_client, PROJECT_NAME, EXPERIMENT_NAME, timeout=1800, quiet=False, stream_logs=False)
```

Let's add the WANDB variables as environment variables to our experiment! This way we can connect to Weight and Biases from our training script.


```python
wandb_api_key_environment_variable = ubiops.EnvironmentVariableCreate(
    name="WANDB_API_KEY", value=WANDB_API_KEY, secret=True
)

wandb_project_environment_variable = ubiops.EnvironmentVariableCreate(
    name="WANDB_PROJECT", value=WANDB_PROJECT, secret=False
)

wandb_entity_environment_variable = ubiops.EnvironmentVariableCreate(
    name="WANDB_ENTITY", value=WANDB_ENTITY, secret=False
)

core_instance.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name="training-base-deployment",
    version=EXPERIMENT_NAME,
    data=wandb_api_key_environment_variable,
)

core_instance.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name="training-base-deployment",
    version=EXPERIMENT_NAME,
    data=wandb_project_environment_variable,
)

core_instance.deployment_version_environment_variables_create(
    project_name=PROJECT_NAME,
    deployment_name="training-base-deployment",
    version=EXPERIMENT_NAME,
    data=wandb_entity_environment_variable,
)
```

With everything set up , we can start sending training jobs to our model!


```python
# Define the input parameters
data_experiments = [
    {"batch_size": 32, "nr_epochs": 5, "learning_rate": 5e-6, "weight_decay": 0.01},
    {"batch_size": 32, "nr_epochs": 5, "learning_rate": 2e-5, "weight_decay": 0.005},
    {"batch_size": 16, "nr_epochs": 5, "learning_rate": 2e-5, "weight_decay": 0.005},
]

# Initiate three training runs using the input parameters
run_ids = []
for index, data_experiment in enumerate(data_experiments):
    new_run = training_instance.experiment_runs_create(
        project_name=PROJECT_NAME,
        experiment_name=EXPERIMENT_NAME,
        data=ubiops.ExperimentRunCreate(
            name=f"training-run-{index}",
            description=f'Trying out a run with {data_experiment ["nr_epochs"]} epochs, batch size {data_experiment["batch_size"]}, learning rate {data_experiment["learning_rate"]} and weight decay {data_experiment ["weight_decay"]}.',
            training_code="training_code/train.py",
            parameters=data_experiment
        ),
        timeout=14400
    )
    run_ids.append(new_run.id)
        
for run_id in run_ids:
    ubiops.utils.wait_for_experiment_run(core_instance.api_client, PROJECT_NAME, EXPERIMENT_NAME, run_id, timeout=1800, quiet=False, stream_logs=False)
```

We can now head to [wandb.ai](https://wandb.ai/home), go to our project, and monitor our results! We can check that our models run on CPUs, and monitor metrics after each epoch!

Using this information, we can select the model with the highest final validation accuracy that we would like to save to the W&B model registry, and deploy to UbiOps.

We can also do this in a more automated way, using the W&B API to identify the best model, link it to the W&B model registry, and then deploy it to UbiOps for inference.


```python
import wandb


wandb_api = wandb.Api()
# Download the best model from our best run based on the final val_accuracy
best_training_run = wandb_api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", order="-summary_metrics.val_accuracy")[0].name
best_model = wandb_api.artifact(
    name=f"{WANDB_ENTITY}/{WANDB_PROJECT}/model-{best_training_run}:latest", type="model"
)
print(f"This is the best training run: {best_training_run}")
```

Now that we have identified the best performing training run in our experiments, let's log the model from that experiment to the Weights & Biases model registry. We can also give the model version an alias of "production", that reflects the phase of the lifecycle the model is in, and can also be used to for automated deployments to our UbiOps inference pipeline.


```python
with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT) as run:
    run.link_artifact(best_model, f"{WANDB_ENTITY}/model-registry/Financial Classifier", aliases=["production"])
```

Next we are going to deploy the model and create an inference endpoint on UbiOps . This is called a *Deployment* in UbiOps and contains the following Python code. The Python code is again executed in an environment with the proper dependencies loaded. For this deployment we will use the same environment as before. We use the initialization function of our deployment to grab our latest model from the W&B model registry and to load it in memory. The request function is used to classify a new input text using the three classes *positive*, *negative* and *neutral*.


```python
!mkdir inference_deployment_package
```


```python
%%writefile inference_deployment_package/deployment.py
import os

import tensorflow as tf
import wandb

from transformers import AutoTokenizer


class Deployment:
    def __init__(self):
        print("Initialising deployment")

        wandb_entity = os.getenv("WANDB_ENTITY")
        # Download the model version aliased 'production' from the W&B model registry and pass reference to load_model
        wandb_api = wandb.Api()
        artifact_obj = wandb_api.artifact(f"{wandb_entity}/model-registry/Financial Classifier:production")
        artifact_path = "artifact_folder"
        artifact_obj.download(artifact_path)

        self.finbert = tf.keras.models.load_model(artifact_path)
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    def request(self, data):
        print("Processing request")

        input = self.tokenizer(text=data["text"], return_tensors="tf")

        output = self.finbert(input)
        output = tf.math.softmax(output["logits"], axis=-1)

        prediction = {
            "positive": float(output[0][0]),
            "negative": float(output[0][1]),
            "neutral": float(output[0][2]),
        }

        # Here we set our output parameters in the form of a json
        return {"prediction": prediction}
```


```python
shutil.make_archive(
    "inference_deployment_package", "zip", ".", "inference_deployment_package"
)
```


```python
inference_deployment_template = ubiops.DeploymentCreate(
    name=INFERENCING_DEPLOYMENT_NAME,
    description="A deployment to label stock headlines by financial sentiment.",
    input_type="structured",
    output_type="structured",
    input_fields=[{"name": "text", "data_type": "string"}],
    output_fields=[{"name": "prediction", "data_type": "dict"}],
)

inference_deployment = core_instance.deployments_create(
    project_name=PROJECT_NAME, data=inference_deployment_template
)
```

We add the WANDB API Token and entity so that our deployment can grab the model from our model registry.


```python
core_instance.deployment_environment_variables_create(
    PROJECT_NAME,
    INFERENCING_DEPLOYMENT_NAME,
    data=wandb_api_key_environment_variable,
)

core_instance.deployment_environment_variables_create(
    PROJECT_NAME,
    INFERENCING_DEPLOYMENT_NAME,
    data=wandb_entity_environment_variable,
)
```

We set up a CPU instance for our inferencing pipeline.


```python
version_template = ubiops.DeploymentVersionCreate(
    version=INFERENCING_DEPLOYMENT_VERSION,
    environment=ENVIRONMENT_NAME,
    instance_type_group_name="2048 MB + 0.5 vCPU",
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=1800,  # = 30 minutes
    request_retention_mode="full",  # input/output of requests will be stored
    request_retention_time=3600,  # requests will be stored for 1 hour
)

version = core_instance.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=INFERENCING_DEPLOYMENT_NAME,
    data=version_template,
)
```

Then we upload our code, finalizing the model deployment.


```python
file_upload_result = core_instance.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=INFERENCING_DEPLOYMENT_NAME,
    version=INFERENCING_DEPLOYMENT_VERSION,
    file="inference_deployment_package.zip",
)

ubiops.utils.wait_for_revision(
    client=api_client,
    project_name=PROJECT_NAME,
    deployment_name=INFERENCING_DEPLOYMENT_NAME,
    version=INFERENCING_DEPLOYMENT_VERSION,
    revision_id=file_upload_result.revision,
    stream_logs=True,
)
```

We can now request our model using its API endpoint!


```python
TEXT = "Stocks rallied and the British pound gained nothing."

request = core_instance.deployment_version_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=INFERENCING_DEPLOYMENT_NAME,
    version=INFERENCING_DEPLOYMENT_VERSION,
    data={"text": TEXT},
)

ubiops.utils.wait_for_deployment_version_request(
    core_instance.api_client, PROJECT_NAME, INFERENCING_DEPLOYMENT_NAME, INFERENCING_DEPLOYMENT_VERSION, request.id,
    timeout=1800, quiet=False, stream_logs=True,
)

print(f"Predictions are: {request.result[f'prediction']}")
```

So that’s it! We have used the training insights from Weights & Biases, and the compute resources and deployment possibilities from UbiOps to create a live and scalable model.

We can reach our model via its API endpoint, when we provide the correct authentication credentials. After setting up the baseline model, you can easily add new deployment versions and tweak the scaling settings. You can scale down to zero in the development phase, and scale up if you want to be able to run multiple inference jobs in parallel! We can actively monitor when and how often our model was requested using the monitoring tabs.
Do you want to try out this workflow for your own training runs, yourself? Feel free to sign up via [ubiops.com](https://app.ubiops.com/sign-up).

