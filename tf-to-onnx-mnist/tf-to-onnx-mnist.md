# Convert your MNIST model from Tensorflow to ONNX and run it on UbiOps twice as fast
[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/tf-to-onnx-mnist/tf-to-onnx-mnist){ .md-button .md-button--primary } [View source code :fontawesome-brands-github:](https://github.com/UbiOps/tutorials/blob/master/tf-to-onnx-mnist/tf-to-onnx-mnist/tf-to-onnx-mnist.ipynb){ .md-button }

ONNX is an open format that is used to represent various Machine Learning models. It can also function as a model compression technique.  In this tutorial we will show you how to convert a Tensorflow based image classification algorithm to ONNX and 
run it on UbiOps using the ONNX runtime. We will show that this allows you to run an inferencing job twice as fast!

First lets connect to UbiOps and load all of our dependencies.


```python
!pip install tensorflow==2.10 tf2onnx==1.13.0 tqdm==4.64.1 'ubiops>=3.12.0' 'protobuf>=3.19.4'
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
    instance_type_group_name='1024 MB + 0.25 vCPU',
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
    instance_type_group_name='1024 MB + 0.25 vCPU',
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

And let's wait until the deployment versions are built until we continue..


```python
ubiops.utils.wait_for_deployment_version(client = api.api_client,
                                   project_name = PROJECT_NAME,
                                   deployment_name = DEPLOYMENT_NAME,
                                   version = "onnx")

ubiops.utils.wait_for_deployment_version(client = api.api_client,
                                   project_name = PROJECT_NAME,
                                   deployment_name = DEPLOYMENT_NAME,
                                   version = "tf")

print("Deployments are ready")

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

Now we create one large batch requests that we can send to the deployment in one go


```python
batch_request_data = []
for image_file in tqdm(filenames):
    # First upload the image
    file_uri = ubiops.utils.upload_file(client, PROJECT_NAME, image_file)
    # Make a request using the file URI as input.
    data = {'image': file_uri}
    batch_request_data.append(data)

```


```python
requests_onnx = api.batch_deployment_version_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version="onnx",
    data=batch_request_data
)
requests_onnx_ids = [request_onnx.id for request_onnx in requests_onnx]

requests_tf = api.batch_deployment_version_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version="tf",
    data=batch_request_data
)
requests_tf_ids = [request_tf.id for request_tf in requests_tf]

```

And then we wait until all requests are finished..


```python
import time

while True:
    requests_onnx = api.deployment_version_requests_batch_get(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        version="onnx",
        data=requests_onnx_ids
    )

    requests_tf = api.deployment_version_requests_batch_get(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        version="tf",
        data=requests_tf_ids
    )

    # Calculate the percentage of completed requests
    onnx_completed_pct = sum(req.status == "completed" for req in requests_onnx) / len(requests_onnx) * 100 if requests_onnx else 0
    tf_completed_pct = sum(req.status == "completed" for req in requests_tf) / len(requests_tf) * 100 if requests_tf else 0

    print(f"ONNX Completed Percentage: {onnx_completed_pct:.2f}%")
    print(f"TensorFlow Completed Percentage: {tf_completed_pct:.2f}%")

    if onnx_completed_pct == 100 and tf_completed_pct == 100:
        break

    time.sleep(1)

```

## Comparing the results

Now that the request are finished we can look at the results. You can do that either by looking at the 'Metrics' tab of 
the UbiOps webappby running the following piece of code.

Note that it can take up to two minutes before metrics become available through our API. So might be required to sleep a 
bit more:


```python
time.sleep(60)
```


```python
#First get the version ids so that we can filter the relevant metrics

tf_version_id = api.deployment_versions_get(PROJECT_NAME,DEPLOYMENT_NAME, "tf").id
onnx_version_id = api.deployment_versions_get(PROJECT_NAME,DEPLOYMENT_NAME, "onnx").id
print(f"Tensorflow deployment version id: {tf_version_id}")
print(f"ONNX deployment version id: {onnx_version_id}")
```


```python
tf_time_series = api.time_series_data_list(
    project_name=PROJECT_NAME,
    metric = "deployments.request_duration",
    start_date=str((datetime.today()- timedelta(days=1)).isoformat()),
    end_date=str(datetime.today().isoformat()),
    aggregation_period = 60*60*24, # seconds/day
    labels = f"deployment_version_id:{tf_version_id}"
)
print(f"Average Tensorflow request duration: {tf_time_series.data_points[-1].value}s ")

onnx_time_series = api.time_series_data_list(
    project_name=PROJECT_NAME,
    metric = "deployments.request_duration",
    start_date=str((datetime.today()- timedelta(days=1)).isoformat()),
    end_date=str(datetime.today().isoformat()),
    aggregation_period = 60*60*24, # seconds/day
    labels = f"deployment_version_id:{onnx_version_id}"
)

print(f"Average ONNX request duration :{onnx_time_series.data_points[-1].value}s")

```

# Cleaning up




```python
api.client_close()
```
