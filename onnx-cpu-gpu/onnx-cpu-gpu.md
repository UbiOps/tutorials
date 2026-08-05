# ONNX models on CPU and GPU

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/onnx-cpu-gpu/onnx-cpu-gpu){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/onnx-cpu-gpu/onnx-cpu-gpu/onnx-cpu-gpu.ipynb){ .md-button .md-button--secondary }

```python
API_TOKEN = 'Token ' # Fill in your token here
PROJECT_NAME = ' '   # Fill in your project name here
DEPLOYMENT_NAME = 'onnx-cpu-gpu'
IMPORT_LINK = "https://storage.googleapis.com/ubiops/deployment_exports/onnx-cpu-gpu-export.zip"
import shutil
import ubiops
import urllib.request 
import random
import glob
import time
from tqdm import tqdm
from datetime import datetime, timedelta

configuration = ubiops.Configuration(host="https://api.ubiops.com/v2.1")
configuration.api_key['Authorization'] = API_TOKEN

client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
api.service_status()

```

## Getting the models on UbiOps


```python
skip_confirmation = True # bool  (optional)

# Create an import
api_response = api.imports_create(PROJECT_NAME, import_link=IMPORT_LINK, skip_confirmation=skip_confirmation)
print(api_response)

```

## Benchmarking


```python
# Download and unpack test images.

urllib.request.urlretrieve("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", "imagenette2-320.tgz")
shutil.unpack_archive("imagenette2-320.tgz", "./") 

```


```python
# Take a random selection of 100 images.

pattern = "imagenette2-320/val/*/*.JPEG" # (or "*.*")
filenames = random.choices(glob.glob(pattern),k=100)
print(len(filenames))
```


```python
# Actual benchmarking

ready = False
while not ready:   # See if deployments are ready
    time.sleep(5)
    response = api.deployment_versions_list(project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME)
    statuses = [d.status == 'available' for d in response]
    ready = all(statuses)
    
    print("Deployments are NOT ready")

print("Deployments are ready")


print("Uploading test images and making requests")
data = []

# We are sending all images in one big batch request
for image_file in tqdm(filenames):    
    # First upload the image
    file_uri = ubiops.utils.upload_file(client, PROJECT_NAME, image_file)
    
    # Make a request using the file uri as input.
    data.append({'image': file_uri})
    
time.sleep(.05) # Let's not crash the api    
api.batch_deployment_version_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version="gpu",
    data=data
)

api.batch_deployment_version_requests_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version="cpu",
    data=data
)

print("Done")
```

Now go to the UbiOps logging page and take a look at the logs of both deployments. You should see a number printed in the logs. This is the average time that an inference takes. After that you can compare it to the following. This code will show the average request time. Note that this is different from each other. the average request time will also include overhead like downloading and uploading images


```python

version_id = api.deployment_versions_get(PROJECT_NAME,DEPLOYMENT_NAME, "cpu").id

print("Average request time (s)")

api_response = api.metrics_get(
    project_name=PROJECT_NAME,
    object_type="deployment_version",
    object_id=version_id,
    metric="compute",
    interval="day",
    start_date=str((datetime.today()- timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')),
    end_date=str(datetime.today().strftime('%Y-%m-%dT%H:%M:%SZ')),
)
print(f"CPU: {api_response[-1].value}")

version_id = api.deployment_versions_get(PROJECT_NAME,DEPLOYMENT_NAME, "gpu").id


api_response = api.metrics_get(
    project_name=PROJECT_NAME,
    object_type="deployment_version",
    object_id=version_id,
    metric="compute",
    interval="day",
    start_date=str((datetime.today()- timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')),
    end_date=str(datetime.today().strftime('%Y-%m-%dT%H:%M:%SZ')),
)
print(f"GPU: {api_response[-1].value}")
```

## Cleaning up


```python
# Close the connection
client.close()
```


```python

```


```python

```
