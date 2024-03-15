# Azure Machine Learning - UbiOps integration

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/azure-machine-learning/azure-machine-learning){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/azure-machine-learning/azure-machine-learning/azure-machine-learning.ipynb){ .md-button .md-button--secondary }

On this page we will show you:

- How to train a model on Azure ML
- How to deploy that model on UbiOps

For this example we will train a model on the MNIST dataset with Azure ML services and then deploy the trained model on UbiOps. Parts of this page were directly taken from https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-train-models-with-aml, which can be found as another notebook here: https://github.com/Azure/MachineLearningNotebooks/blob/master/tutorials/image-classification-mnist-data/img-classification-part1-training.ipynb. 
The trained model can be adapted to your usecase. The MNIST model is taken merely to illustrate how a model trained with Azure ML services could be converted to run on UbiOps. 


If you want to [download](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/azure-machine-learning/azure-machine-learning){:target="_blank"} and run this notebook, make sure that it is running from an environment with the requirements (see requirements.txt) installed.
Also provide the Azure config.json in the `config` folder and ensure to fill in the configuration below.

## Download the necessary files


```python
import requests

config = requests.get('https://storage.googleapis.com/ubiops/data/Integration%20with%20cloud%20provider%20tools/azure-ml/config.json')
deployment_package = requests.get('https://storage.googleapis.com/ubiops/data/Integration%20with%20cloud%20provider%20tools/azure-ml/deployment_package.zip')
training_files = requests.get('https://storage.googleapis.com/ubiops/data/Integration%20with%20cloud%20provider%20tools/azure-ml/training_files.zip')
requirements = requests.get('https://storage.googleapis.com/ubiops/data/Integration%20with%20cloud%20provider%20tools/azure-ml/requirements.txt')

with open("config.json", "wb") as f:
    f.write(config.content)

with open('deployment_package.zip', 'wb') as f:
  f.write(deployment_package.content)

with open('training_files.zip', 'wb') as f: 
    f.write(training_files.content)

with open('requirements.txt', 'wb') as f: 
    f.write(requirements.content)

```


```python
## UbiOps configuration
API_TOKEN = '<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>'
PROJECT_NAME= '<INSERT PROJECT NAME IN YOUR ACCOUNT>'
DEPLOYMENT_NAME='mnist'
DEPLOYMENT_VERSION='v1'
```


```python
## Azure configurations
# You can keep the default values in this configuration, or adjust according to your own use case
workspace_config_file = "config.json"
experiment_name = "sklearn-mnist"
model_name = "sklearn_mnist"
compute_name = "aml-compute"
vm_size = "STANDARD_D2_V2"
mnist_dataset_name="sklearn-mnist-opendataset"
env_name="sklearn-mnist-env"
```


```python
!pip install ubiops
!pip install azureml-core
!pip install azureml-opendatasets
```


```python
import os 
import shutil
import ubiops

from azureml.core import Experiment, Workspace, Datastore, Dataset, ScriptRunConfig, Model
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.opendatasets import MNIST

from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
```


```python
# Configure the Workspace and create an Azure Experiment
# Running this cell will open a new window in which you are asked to log into your Azure account
ws = Workspace.from_config(workspace_config_file)
experiment = Experiment(workspace=ws, name=experiment_name)
```


```python
# Find compute target with name {compute_name} or create one
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    
    if compute_target and type(compute_target) is AmlCompute:
        print('Found compute target:', compute_name)
else:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                              min_nodes=0,
                                                              max_nodes=1)
    # Create the compute target
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

    # For a more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())

```

### Retrieving Data
We now have compute resources to train our model in the cloud. 
The next step is retrieving data.


```python
# Create a folder and store the MNIST dataset in it
data_folder = os.path.join(os.getcwd(), 'data')
os.makedirs(data_folder, exist_ok=True)

mnist_file_dataset = MNIST.get_file_dataset()

try:
    mnist_file_dataset.download(data_folder, overwrite=False,)
except RuntimeError:
    # File already exists
    pass

# Register the data to the workspace
mnist_file_dataset = mnist_file_dataset.register(
    workspace=ws,
    name=mnist_dataset_name,
    description='Train and test dataset',
    create_new_version=False
)
```

### Training a model
The next step is to configure the training job. For this, we first create a virtual environment in our Workspace which holds all the required packages. Then we upload the training script that was created with Azure ML services. Just like with the data, we store it in a folder that's registered to the workspace. We already have created the files for training and stored them in the folder `training_files`. 
Lastly, we configure and submit the job.


```python
# Install required packages
env = Environment(env_name)
cd = CondaDependencies.create(
    pip_packages=['azureml-dataprep[pandas,fuse]>=1.1.14', 'azureml-defaults'], 
    conda_packages = ['scikit-learn==0.22.1']
)

env.python.conda_dependencies = cd

# Register environment to re-use later
env.register(workspace = ws)
```


```python
# Register the training_files directory to the workspace
script_folder = os.path.join(os.getcwd(), "training_files")


# Give the specification of the job...
args = ['--data-folder', mnist_file_dataset.as_mount(), '--regularization', 0.5]

src = ScriptRunConfig(source_directory=script_folder,
                      script='train.py', 
                      arguments=args,
                      compute_target=compute_target,
                      environment=env)
# ..and run! 
run = experiment.submit(config=src)
run
```


```python
# The status of the job will initialize with 'Starting', but will transit into 'Queued', 'Running' and finally 'Completed'. 
# However, completing the first run can take up to 10 minutes.

# Don't cancel the jupyter cell!
run.wait_for_completion(show_output=False)  # Specify True for a verbose log
```


```python
# Curious about the accuracy on the test set?
print(run.get_metrics())
```


```python
# Register the model
model = run.register_model(model_name=model_name,
                           model_path='outputs/sklearn_mnist_model.pkl')
print(model.name, model.id, model.version, sep='\t')
```

## Deploying a with Azure ML trained model to Ubiops
The last step in the training script wrote the model file to `sklearn_mnist_model.pkl` to a directory named `outputs` in the VM of the cluster where the job is run. We can pick it up from there and make it ready for use in UbiOps in a few simple steps.


```python
# Download model file to a deployment package
if not os.path.exists('deployment_package/sklearn_mnist_model.pkl'):
    model_path = Model(ws,'sklearn_mnist').download('deployment_package')
```


```python
# Connect to UbiOps
client = ubiops.ApiClient(ubiops.Configuration(api_key={'Authorization': API_TOKEN}, 
                                               host='https://api.ubiops.com/v2.1'))
api = ubiops.CoreApi(client)
api.service_status()
```


```python
# Create the MNIST deployment
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description='MNIST model trained with Azure ML',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'image', 'data_type':'file'}
    ],
    output_fields=[
        {'name':'prediction', 'data_type':'int'}
    ],
    labels={'demo': 'azure-ml'}
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)

# Create a version for the deployment
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-9',
    instance_type='1024mb',
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=1800, # = the model will wait for 30 minutes after the last request
    request_retention_mode='full', # input/output of requests will be stored
    request_retention_time=3600 # requests will be stored for 1 hour
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=version_template
)

# Upload a zipped deployment package
file_upload_result =api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file=shutil.make_archive(f"deployment_package", 'zip', '.', "deployment_package")
)

# Status of the version will be building
version_status = api.deployment_versions_get(       
    project_name=PROJECT_NAME,        
    deployment_name=DEPLOYMENT_NAME,        
    version=DEPLOYMENT_VERSION    
)  

version_status.status
```

The above cell creates a deployment and version on UbiOps. Creating a deployment lets you define the in- and output of your model, allowing UbiOps to check if the data that is coming in or out is of the correct type. With the version details, you can adapt the configuration of the serving of your model. Uploading a deployment package, triggered the build, where UbiOps checks if it’s able to serve your model.

## Making a request
That's it! Now we can make requests to our model. For your convenience, we've already extracted some test images from the MNIST dataset, you can download them using [this link](https://storage.googleapis.com/ubiops/data/Integration%20with%20cloud%20provider%20tools/azure-ml/test_images.zip). From here, we can loop over the images, upload them to UbiOps and use them in a direct or batch request. Alternatively, you can now switch to our user interface and make a request with the images manually.




```python
# Upload the images to UbiOps and save the file uri's
files_list = []

for image in os.listdir(os.path.join(os.getcwd(), 'test_images')):
    image_path = os.path.join(os.getcwd(), 'test_images', image)
    file_uri = ubiops.utils.upload_file(
        client=client,
        project_name=PROJECT_NAME,
        file_path=image_path
    )
    data = {'image': file_uri}
    files_list.append(data)
files_list
```


```python
# Make one batch request (consisting of a request per uploaded file) to our deployment
# The response is a list of all the batch requests we created
response = api.batch_deployment_version_requests_create(
    project_name=PROJECT_NAME, 
    deployment_name=DEPLOYMENT_NAME, 
    version=DEPLOYMENT_VERSION, 
    data=files_list
)
response
```


```python
# Let's see what the result is of one of our requests. The request will be initialized with status 'pending',
# after which it will turn into 'processing' and finally 'completed'.
api.deployment_version_requests_batch_get(
    project_name=PROJECT_NAME, 
    deployment_name=DEPLOYMENT_NAME, 
    version=DEPLOYMENT_VERSION, 
    request_id=response[0].id)

#Re-run this cell untill the status is 'completed' and see the result of your request!
```

## All done! Let's close the client properly.


```python
client.close()
```
**Note**: This notebook runs on Python 3.9 and uses UbiOps Client Library 3.15.0.

## Exploring further
You can go ahead to the Web App and take a look in the user interface at what you have just built. If you want you can create a request to the pipeline with empty input, to see what happens.

So there we have it! We have made a model with Azure ML and deployed it on UbiOps, making integration of the two services very easy. You can use this notebook as a reference for your own model and use case. Just adapt the code in the deployment package and alter the input and output fields as you wish and you should be good to go. 

For any questions, feel free to reach out to us via the customer service portal: https://ubiops.atlassian.net/servicedesk/customer/portals
