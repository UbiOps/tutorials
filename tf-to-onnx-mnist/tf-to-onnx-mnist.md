# Convert your MNIST model from Tensorflow to ONNX and run it on UbiOps twice as fast

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/tf-to-onnx-mnist/tf-to-onnx-mnist){ .md-button .md-button--primary } [View source code :fontawesome-brands-github:](https://github.com/UbiOps/tutorials/blob/master/tf-to-onnx-mnist/tf-to-onnx-mnist/tf-to-onnx-mnist.ipynb){ .md-button }


ONNX is an open format that is used to represent various Machine Learning models. It can also function as a model compression technique.  In this tutorial we will show you how to convert a Tensorflow based image classification algorithm to ONNX and 
run it on UbiOps using the ONNX runtime. We will show that this allows you to run an inferencing job twice as fast!

First lets connect to UbiOps and load all of our dependencies.


```python
!pip install tensorflow==2.10 tf2onnx==1.13.0 tqdm==4.64.1 ubiops>=3.12.0 protobuf>=3.19.4
```

First connect to our API


```python
API_TOKEN = 'Token ' # Fill in your token here
PROJECT_NAME = ''    # Fill in your project name here
DEPLOYMENT_NAME = 'tf-vs-onnx-test'
import ubiops 
import shutil
import random, glob
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import shutil


configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key['Authorization'] = API_TOKEN

client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
api.service_status()
```

## Converting the model

We first download an h5 model from our public online bucket, then convert it as a `SavedModel`. Lastly, we convert it to an `onnx` model using the `tf2onnx` package.

If everything worked correctly you should have the ONNX model at ```mnist_deployment_onnx_package/mnist.onnx```.


```python
import os
import urllib.request
import zipfile
from tensorflow.keras.models import load_model

#Get bucket from online repo
bucket_name = "ubiops"
file_path = "demo-helper-files/cnn.zip"

# Create the URL for the file
url = f"https://storage.googleapis.com/{bucket_name}/{file_path}"

#Write zipfile to cnn folder
urllib.request.urlretrieve(url, "cnn")

#write modelfile to cnn_dir folder
with zipfile.ZipFile("cnn", 'r') as zip_ref:
    zip_ref.extractall('cnn_dir')

model = load_model("cnn_dir/cnn.h5")

#Save as a SavedModel to the mnist_model directory
!mkdir mnist_model
model.save("mnist_model")
```

## Preparing the comparison

The next step is to create two deployments. One with the original Tensorflow based runtime and the second with the ONNX model runnning on the ONNX runtime.

The following code will save the Tensorflow model, the requirements.txt's and the deployments.py's to the mnist_deployment_package directory.


```python
!mkdir mnist_deployment_package
```


```python
#Copy the tensorflowmodel to the deployment package
import shutil
shutil.copy('cnn_dir/cnn.h5', 'mnist_deployment_package/cnn.h5')
```


```python
%%writefile ./mnist_deployment_package/requirements.txt

# This file contains package requirements for the deployment
# installed via PIP. Installed before deployment initialization
tensorflow==2.10
imageio==2.26.0
h5py==3.8.0
numpy==1.24.1
Pillow==9.4.0

```


```python
%%writefile ./mnist_deployment_package/deployment.py
"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
from tensorflow.keras.models import load_model
from imageio import imread
import numpy as np


class Deployment:

    def __init__(self, base_directory, context):

        print("Initialising deployment")

        weights = os.path.join(base_directory, "cnn.h5")
        self.model = load_model(weights)

    def request(self, data):

        print("Processing request")

        x = imread(data['image'])
        # convert to a 4D tensor to feed into our model
        x = x.reshape(1, 28, 28, 1)
        x = x.astype(np.float32) / 255

        out = self.model.predict(x)

        # here we set our output parameters in the form of a json
        return {'prediction': int(np.argmax(out)), 'probability': float(np.max(out))}

```

Now build a deployment package that hosts the ONNX model


```python
!mkdir mnist_deployment_onnx_package
```


```python
#Convert the model from SavedModel format to onnx, and store inside the ONNX deployment package
!python3 -m tf2onnx.convert --saved-model mnist_model --opset 13 --output mnist_deployment_onnx_package/mnist.onnx
```


```python
%%writefile ./mnist_deployment_onnx_package/deployment.py
"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
import onnxruntime as rt
from imageio import imread
import numpy as np


class Deployment:

    def __init__(self, base_directory, context):
        self.sess = rt.InferenceSession("mnist.onnx")
        self.input_name = self.sess.get_inputs()[0].name

    def request(self, data):


        x = imread(data['image'])
        # convert to a 4D tensor to feed into our model
        x = x.reshape(1, 28, 28, 1) 
        x = x.astype(np.float32) / 255

        print("Prediction being made")

        prediction = self.sess.run(None, {self.input_name: x})[0]

        return {'prediction': int(np.argmax(prediction)), 'probability': float(np.max(prediction))}

       

```


```python
%%writefile ./mnist_deployment_onnx_package/requirements.txt

# This file contains package requirements for the deployment
# installed via PIP. Installed before deployment initialization

onnx==1.12.0
onnxruntime==1.12.0
imageio==2.26.0
numpy==1.24.1
```

Now that the deployment packages are created, you can upload them to UbiOps. We will make one deployment with two versions, one running Tensorflow while the other is running ONNX.


```python
mnist_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description='A deployment to classify handwritten digits.',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name': 'image', 'data_type': 'file'}
    ],
    output_fields=[
        {'name': 'prediction', 'data_type': 'int'},
        {'name': 'probability', 'data_type': 'double'}
    ]
)

mnist_deployment = api.deployments_create(project_name=PROJECT_NAME, data=mnist_template)
print(mnist_deployment)
```


```python
version_template = ubiops.DeploymentVersionCreate(
    version="onnx",
    environment='python3-10',
    instance_type='1024mb',
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='full',  # input/output of requests will be stored
    request_retention_time=3600  # requests will be stored for 1 hour
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=version_template
)
```


```python
# Zip the deployment package
shutil.make_archive('mnist_deployment_onnx_package', 'zip', '.', 'mnist_deployment_onnx_package')


upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version="onnx",
    file='mnist_deployment_onnx_package.zip'
)
print(upload_response)
```


```python
version_template = ubiops.DeploymentVersionCreate(
    version="tf",
    environment='python3-10',
    instance_type='1024mb',
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='full',  # input/output of requests will be stored
    request_retention_time=3600  # requests will be stored for 1 hour
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=version_template
)
```


```python
# Zip the deployment package
shutil.make_archive('mnist_deployment_package', 'zip', '.', 'mnist_deployment_package')


upload_response = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version="tf",
    file='mnist_deployment_package.zip'
)
print(upload_response)
```

## Benchmarking

If everything went well there should now be a deployment in UbiOps with two versions. We can now compare the average request time by sending both versions a list of 100 images (one image per request.)


```python
import urllib
import zipfile
#Get dummy data from our online bucket
bucket_name = "ubiops"
file_path = "demo-helper-files/mnist_png.zip"

# Create the URL for the file
url = f"https://storage.googleapis.com/{bucket_name}/{file_path}"

urllib.request.urlretrieve(url, "mnist_png.zip")

with zipfile.ZipFile("mnist_png.zip", 'r') as zip_ref:
    zip_ref.extractall('./')
```


```python
pattern = "mnist_png/testing/*/*.png" # (or "*.*")
filenames = random.choices(glob.glob(pattern),k=100)
print(filenames)
```


```python
ready = False
while not ready:   
    time.sleep(5)
    response = api.deployment_versions_list(project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME)
    statuses = [d.status == 'available' for d in response]
    ready = all(statuses)
    
    print("Deployments are NOT ready")

print("Deployments are ready")

print("Uploading test images and making requests")

for image_file in tqdm(filenames):    
    # First upload the image
    file_uri = ubiops.utils.upload_file(client, PROJECT_NAME, image_file)
    
    # Make a request using the file URI as input.
    data = {'image': file_uri}
    
    time.sleep(.05) # Let's not crash the api    
    api.deployment_version_requests_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        version="onnx",
        data=data
    )

    api.deployment_version_requests_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        version="tf",
        data=data
    )

print("Done")
```

## Comparing the results

Now that the request are finished we can look at the results. You can do that either by looking at the 'Metrics' tab of the UbiOps webappby running the following piece of code.


```python

version_id = api.deployment_versions_get(PROJECT_NAME,DEPLOYMENT_NAME, "tf").id

print("Average request time (s)")

api_response = api.metrics_get(
    project_name=PROJECT_NAME,
    object_type="deployment_version",
    object_id=version_id,
    metric="compute",
    interval="day",
    start_date=str((datetime.today()- timedelta(days=1)).isoformat()),
    end_date=str(datetime.today().isoformat()),
)
print(f"Tensorflow: {api_response[-1].value}")

version_id = api.deployment_versions_get(PROJECT_NAME,DEPLOYMENT_NAME, "onnx").id


api_response = api.metrics_get(
    project_name=PROJECT_NAME,
    object_type="deployment_version",
    object_id=version_id,
    metric="compute",
    interval="day",
    start_date=str((datetime.today()- timedelta(days=1)).isoformat()),
    end_date=str(datetime.today().isoformat())
)
print(f"ONNX:       {api_response[-1].value}")
```

# Cleaning up




```python
api.client_close()
```
