# Retrain ResNet using PyTorch


[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/retrain-resnet-pytorch/retrain-resnet-pytorch){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/retrain-resnet-pytorch/retrain-resnet-pytorch/retrain-resnet-pytorch.ipynb){ .md-button .md-button--secondary }

In this example, we show how to retrain a PyTorch model on UbiOps. In this end-to-end example, we first set-up an `Environment` in which our training jobs can run. Then we define a `train.py` script that we can apply to our `Environment`. The training script imports ResNet with pretrained weights, and retrains that model on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Finally, we benchmark its performance on the test set, and add that as a `metric` to our output. Snippets from this workflow can be used to retrain your own models.

Let us first install the UbiOps Python client.


```python
!pip install "ubiops >= 3.15"
```

# 1) Set project variables and initialize the UbiOps API Client
First, make sure you create an API token with project editor permissions in your UbiOps project and paste it below. Also fill in your corresponding UbiOps project name.


```python
from datetime import datetime
import yaml
import os
import ubiops

dt = datetime.now()

API_TOKEN = 'Token '   # Paste your API token here. Don't forget the `Token` prefix
PROJECT_NAME = ''  # Fill in the corresponding UbiOps project name

configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key['Authorization'] = API_TOKEN

api_client = ubiops.ApiClient(configuration)
core_instance = ubiops.CoreApi(api_client=api_client)
training_instance = ubiops.Training(api_client=api_client)
print(core_instance.service_status())
```

Set-up a training instance in case you have not done this yet in your project. This action will create a base training deployment, that is used to host training experiments.


```python
training_instance = ubiops.Training(api_client=api_client)
try:
    training_instance.initialize(project_name=PROJECT_NAME)
except ubiops.exceptions.ApiException as e:
    print(f"The training feature may already have been initialized in your project:\n{e}")

```

## Defining the code environment
Our training code needs an environment to run in, with a specific Python language version, and some dependencies, like `PyTorch`. You can create and manage environments in your UbiOps project. We create an environment named `python3-11-pytorch-retraining`, select Python 3.11 and upload a requirements.txt which contains the relevant dependencies.

The environment can be reused and updated for different training jobs (and deployments!). The details of the environment are visible in the 'environments' tab in the UbiOps UI.


```python
training_environment_dir = 'training_environment'
ENVIRONMENT_NAME = 'python3-11-pytorch-retraining'
```


```python
%mkdir {training_environment_dir}

```


```python
%%writefile {training_environment_dir}/requirements.txt
torch==1.13.1
torchvision==0.14.1
```


```python
import shutil 
training_environment_archive = shutil.make_archive(f'{training_environment_dir}', 'zip', '.', f'{training_environment_dir}')

# Create experiment. Your environment is set-up in this step. It may take some time to run.

try:
    api_response = core_instance.environments_create(
        project_name=PROJECT_NAME,
        data=ubiops.EnvironmentCreate(
        name=ENVIRONMENT_NAME,
        #display_name=ENVIRONMENT_NAME,
        base_environment='python3-11',
        description='Training environment with Python 3.11 and PyTorch 1.13 for Resnet retraining',
        )
    )

    core_instance.environment_revisions_file_upload(
        project_name=PROJECT_NAME,
        environment_name=ENVIRONMENT_NAME,
        file=training_environment_archive
    )
except ubiops.exceptions.ApiException as e:
    print(e)
```

## Configure an experiment
The basis for model training in UbiOps is an 'Experiment'. An experiment has a fixed code environment and hardware (instance) definition, but it can hold many different 'Runs'. You can create an experiment in the WebApp or use the client library, as we do here.

This bucket will be used to store your training jobs, model artifacts and any other files that are created during the training run.


```python
EXPERIMENT_NAME = 'retrain-resnet-pytorch' # str
BUCKET_NAME = 'default'

try:
    experiment = training_instance.experiments_create(
        project_name=PROJECT_NAME,
        data=ubiops.ExperimentCreate(
            instance_type_group_name='4096 MB + 1 vCPU',
            description='Retrain the ResNet model on CIFAR-10 data',
            name=EXPERIMENT_NAME,
            environment=ENVIRONMENT_NAME,
            default_bucket= BUCKET_NAME
        )
    )
except ubiops.exceptions.ApiException as e:
    print(e)
```

## Define and start a training run
A training job in UbiOps is called a run. To run Python code for training on UbiOps, we need to create a file named `train.py` and include our training code here. This code will execute as a single 'Run' as part of an 'Experiment' and uses the code environment and instance type (hardware) as defined with the experiment as shown before.  
Let’s take a look at the training script. The `train.py` script requires a `train()` function, with input parameters `training_data` (a file path to your training data) and `parameters` (a dictionary that contains parameters of your choice). More detailed information on the training code format can be found in the [UbiOps training documentation](https://ubiops.com/docs/training/#training-code-format).  
 
In this example, we will download the `CIFAR-10` dataset using the `torchvision` package during the training process, so there is no need to upload our own dataset.

Now that we have our `environment` and `experiment` set-up, it is easy to initiate runs. The `RUN_NAME` and `RUN_SCRIPT` can easily be tweaked in the next two cells, and sent to the relevant `experiment` in the cell after.


```python
RUN_NAME = 'training-run'
RUN_SCRIPT = f'{RUN_NAME}.py'
```


```python
%%writefile {RUN_SCRIPT}
import json
import os

import torch
import torchvision
import time

import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


import torchvision.models as models

class Net(nn.Module):
   def __init__(self):
        super().__init__()
        
        #Preload resnet. Supress logs while importing
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT', progress = False)
        self.loss = nn.CrossEntropyLoss()
        
        #Apply our optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr = 0.01, momentum = 0.9)

   def forward(self, x, target=None):
        x = self.model(x)

        if self.training:
            loss = self.loss(x, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return x, loss
        else:
            return x



def train(training_data, parameters, context):
    
    # Check the availability of a GPU (this tutorial focuses on a CPU instance, 
    # but can be extended to run on a GPU instance)
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get batch size from input parameters
    batch_size = parameters['batch_size']
    epochs = int(parameters['epochs'])
    
    print(f"Unpacked parameters {parameters}")
    # Create data input transformer
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
 
    # Select the dataset from torchvision
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        dataset=testset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last = True,
        num_workers=2
    )

    
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = Net()
    
    net.to(device)
    print(f"Moved Resnet model to {device}")
    
    print("Starting the model training!")
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for _ , data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            _, loss = net(inputs, labels)
            
            # print statistics
            running_loss += loss.item()
            
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.3f}')
              
    print("Finished model training")

    model_path =  "./cifar_net.pth"
    # Return the trained model
    torch.save(net.state_dict(), model_path)
    print(f"Saved model to {model_path} ")
    
    
    print("Evaluating the model performance")    
    testnet = Net()
    testnet.to(device)
    testnet.load_state_dict(torch.load(model_path))
    testnet.eval()
    
    # Test accuracy
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = testnet(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the retrained Resnet model on all 10000 test images: {100 * correct // total} %')
    
    run_id = context['id']
    return {
        "artifact": {
            "file": "cifar_net.pth",
            "bucket": os.environ.get("SYS_DEFAULT_BUCKET", "default"),
            "bucket_file": f"{run_id}/cifar_net.pth"
        },
        "metrics": json.dumps({
            "accuracy": 100 * correct // total
        })
    }

```

Now we initiate the training run. Do note that each epoch takes around 15 minutes to finish on a 4GB CPU instance. For demonstration purposes, we will run 1 epoch only, but feel free to increase this number if you have the time. The workload is running in the cloud, so there is no need to keep your local machine on.


```python
new_run = training_instance.experiment_runs_create(
    project_name=PROJECT_NAME,
    experiment_name=EXPERIMENT_NAME,
    data=ubiops.ExperimentRunCreate(
        name=RUN_NAME,
        description='First try!',
        training_code= RUN_SCRIPT,
        training_data= None,
        parameters={
            'epochs': 1, # example parameters
            "batch_size" : 32
        }
    ),
    timeout=14400
)
```

## Analyse the logs while training
One way to measure our model performance during training is to check the logs. We can do so in the UI, or by using the relevant API endpoint. To format the the logs in a pretty way, we will use the `pprint` library.

```python
import pprint
from datetime import datetime

current_datetime = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

logs = core_instance.projects_log_list(
    project_name = PROJECT_NAME,
    data = {
    "date_range": -86400, # Get results between current_datetime and 86400 seconds before
    "filters": {
        "deployment_name": "training-base-deployment",
        "deployment_request_id": new_run.id, #
        "deployment_version": EXPERIMENT_NAME,
 #       "system": False # Optional filter to enable/disable system-level logs, see docs: "https://ubiops.com/docs/monitoring/logging/#system-logs"
    },
    "limit": 100,
    "date": current_datetime,
})

logs_body = {log.log for log in logs}
pprint.pprint(logs_body, indent = 1)
```



## Wrapping up

So that's it! We have created a set-up where we can retrain ResNet on UbiOps using the PyTorch library. The training script,  model artifact and output metric are stored on UbiOps. This creates a proper basis for improving the accuracy of our final custom model. 

Let us close the connection to the UbiOps API


```python
core_instance.client_close()
```
