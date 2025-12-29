<img src="https://ubiops.com/wp-content/uploads/2020/12/Group-2.svg" title="UbiOps Logo" width=100px/>

# Training a Tensorflow model on UbiOps

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/tensorflow-training/tensorflow-training){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/tensorflow-training/tensorflow-training/tensorflow-training.ipynb){ .md-button .md-button--secondary }

**This tutorial is part of a blogpost.**

**In this notebook, we will show how to run a training job for a Tensorflow model on the UbiOps platform.**

We will define and create a UbiOps training script. Using the UbiOps Python client we will configure the environment in which this script can be run, and an experiment which is used to analyse and track our results

You can try it yourself by using a valid UbiOps API token and project name in the cell below.

##### **About the training code**
The training function we will deploy expects a path to a zipped `training data` file,  `the number of epochs`, and the `batch_size` as input. As output it will give the trained `model artifact` as well as the final `loss` and `accuracy` for the training job.
- The training code and data is based on one of the Tensorflow tutorials for training a model on the 'flowers dataset'. Source: https://www.tensorflow.org/tutorials/load_data/images
- The corresponding URL for the training data archive is: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

| Training code in- and output variables |        |                            |
|---------------------------------------|--------|----------------------------|
|                                       | **Fields (type)**                  | **Keys of dictionary**     |
| **Input fields**                      | training_data (file) |                            |
|                                       | parameters (dict) | {epochs (*data_type=* as integer), batch_size  (*data_type=* integer)} |
| **Output fields**                     | artifact (file) |                  |
|                                       | metrics (dict) | {accuracy (*data_type=* float), loss (*data_type=* float), loss_history (*data_type=* list[float]), acc_history (*data_type=* list[float])}           |


To interface with UbiOps through code we need the UbiOps Python client library. In the following cell it will be installed.


```python
!pip install --upgrade ubiops
```

***
# 1) Set project variables and initialize the UbiOps API Client
First, make sure you create an **API token** with `project editor` permissions in your UbiOps project and paste it below. Also fill in your corresponding UbiOps project name. 


```python
from datetime import datetime

dt = datetime.now()
import yaml
import os
import ubiops

API_TOKEN = "Token "  # Paste your API token here. Don't forget the `Token` prefix
PROJECT_NAME = ""  # Fill in the corresponding UbiOps project name

configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key["Authorization"] = API_TOKEN

api_client = ubiops.ApiClient(configuration)
core_instance = ubiops.CoreApi(api_client=api_client)
training_instance = ubiops.Training(api_client=api_client)
print(core_instance.service_status())
```

## Initiate the training functionallity in your UbiOps project

Set-up a training instance in case you have not done this yet in your project. This action will create a base training deployment, that is used to host training experiments.


```python
training_instance = ubiops.Training(api_client=api_client)
try:
    training_instance.initialize(project_name=PROJECT_NAME)
except ubiops.exceptions.ApiException as e:
    print(
        f"The training feature may already have been initialized in your project:\n{e}"
    )
```

## Defining the code environment

Our training code needs an environment to run in, with a specific Python language version, and some dependencies, like `Tensorflow`. You can create and manage environments in your UbiOps project. 
We create an environment named 'python3-11-tensorflow-training', select Python 3.11 and upload a `requirements.txt` which contains the relevant dependencies.

The environment can be  reused and updated for different training jobs (and deployments!). The details  of the environment are visible in the 'environments' tab in the UbiOps UI.


```python
training_environment_dir = "training_environment"
ENVIRONMENT_NAME = "python3-11-tensorflow-training"
```


```python
%mkdir {training_environment_dir}
```


```python
%%writefile {training_environment_dir}/requirements.txt
numpy==1.24.1
tensorflow==2.10.0
joblib==1.2.0
```

Now zip the environment like you would zip a deployment package, and create an environment


```python
import shutil

training_environment_archive = shutil.make_archive(
    f"{training_environment_dir}", "zip", ".", f"{training_environment_dir}"
)

# Create experiment. Your environment is set-up in this step. It may take some time to run.

try:
    api_response = core_instance.environments_create(
        project_name=PROJECT_NAME,
        data=ubiops.EnvironmentCreate(
            name=ENVIRONMENT_NAME,
            # display_name=ENVIRONMENT_NAME,
            base_environment="python3-11",
            description="Test training environment with tensorflow 2.10 and some helper functions",
        ),
    )

    core_instance.environment_revisions_file_upload(
        project_name=PROJECT_NAME,
        environment_name=ENVIRONMENT_NAME,
        file=training_environment_archive,
    )
except ubiops.exceptions.ApiException as e:
    print(e)
```

## Configure an experiment
The basis for model training in UbiOps is an 'Experiment'. An experiment has a fixed code environment and hardware (instance) definition, but it can hold many different 'Runs'.

You can create an experiment in the WebApp or use the client library, as we're here.

This bucket will be used to store your training jobs and model callbacks. In case you want to continue without creating a bucket, you can use the `default` bucket, that is always present inside your account.


```python
EXPERIMENT_NAME = "training-experiment-demo"  # str
BUCKET_NAME = "default"

try:
    experiment = training_instance.experiments_create(
        project_name=PROJECT_NAME,
        data=ubiops.ExperimentCreate(
            instance_type_group_name="4096 MB + 1 vCPU",
            description="Train test experiment",
            name=EXPERIMENT_NAME,
            environment=ENVIRONMENT_NAME,
            default_bucket=BUCKET_NAME,
        ),
    )
except ubiops.exceptions.ApiException as e:
    print(e)
```

## Load the training data

We will download the publicly available `flower photos` dataset. We will our model on this file.


```python
import urllib.request

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
training_data = "flower_photos.tgz"

urllib.request.urlretrieve(url, training_data)

print(f"File downloaded successfully to '{training_data}'.")
```

We can inspect the dataset by untarring the tarfile, this step is optional


```python
import tarfile

file_dir = "flower_photos"
with tarfile.open(training_data, "r:gz") as tar:
    path = tar.extractall("./")
```

## Define and start a training run

A training job in UbiOps is called a run. To run any Python code on UbiOps, we need to create a file named `train.py` and include our training code here. This code will execute as a single 'Run' as part of an 'Experiment' and uses the code environment and instance type (hardware) as defined with the experiment as shown before.
Let’s take a look at the training script. The UbiOps `train.py` structure is quite simple. It only requires a train() function, with input parameters `training_data` (a file path to your training data) and `parameters`(a dictionary that contains parameters of your choice).  If we upload this training code, along with the `training_data` file and some values for our input parameters, a training run is initiated! You can run different training runs in parallel, with different scripts or different hyperparameters. An example of this set up can be
found in the [XGBoost training tutorial](https://ubiops.com/docs/ubiops_tutorials/xgboost-training/xgboost-training/).


```python
RUN_NAME = "training-run"
RUN_SCRIPT = f"{RUN_NAME}.py"
```


```python
%%writefile {RUN_SCRIPT}
import os
import tensorflow as tf
import joblib
import pathlib
import shutil
import tarfile

def train(training_data, parameters, context = {}):
    '''All code inside this function will run when a call to the deployment is made.'''

    img_height = 180
    img_width = 180
    batch_size = int(parameters['batch_size']) #Specify the batch size
    nr_epochs = int(parameters['nr_epochs']) #Specify the number of epochs
  

    # Load the training data
    extract_dir = "flower_photos"

    with tarfile.open(training_data, 'r:gz') as tar:
      tar.extractall("./")

    data_dir = pathlib.Path(extract_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)


    # Standardize the data
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


    # Train the model
    num_classes = 5

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ]) 

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=nr_epochs    
    )
    
    eval_res = model.evaluate(val_ds)
    
    
    # Return the trained model file and metrics
    joblib.dump(model, 'model.pkl')
    fin_loss = eval_res[0]
    fin_acc = eval_res[1]
    
    print(history)
    print(history.history)
    return {
        "artifact": 'model.pkl',
        "metrics": {'fin_loss' : fin_loss,
                    'fin_acc' : fin_acc,
                    "loss_history": history.history["loss"],
                    "acc_history" : history.history["accuracy"]},
        }

```


```python
new_run = training_instance.experiment_runs_create(
    project_name=PROJECT_NAME,
    experiment_name=EXPERIMENT_NAME,
    data=ubiops.ExperimentRunCreate(
        name=RUN_NAME,
        description="Trying out a first run run",
        training_code=RUN_SCRIPT,
        training_data=training_data,
        parameters={"nr_epochs": 2, "batch_size": 32},  # example parameters
    ),
    timeout=14400,
)
```

We can easily finetune our training code and execute a new training code, and analyse the logs along the way.
When training a model it is important to keep track of the training progress and convergence. We do this by looking at the training loss and accuracy metrics. Packages like Tensorflow will print these for you continuously, and we’re able to track them in the logging page of the UbiOps UI.
If you notice a training job is not converging, you’re able to cancel the request and try it again with different data or different parameters.

Additionaly you can create custom metrics in UbiOps, you can find more information about that [here](https://ubiops.com/docs/monitoring/metrics/#custom-metrics).

## Evaluating the output
When the training runs are completed, the training run will provide you with the trained parameter file, the final accuracy and loss. The parameter file is stored inside a UbiOps bucket. You can easily navigate to this location from the training-run interface.
You can compare metrics of different training runs easily inside the Evaluation page of the Training tab, allowing you to analyze which code or which hyperparameters worked best.

And that’s it, you just trained a Tensorflow model on UbiOps!

