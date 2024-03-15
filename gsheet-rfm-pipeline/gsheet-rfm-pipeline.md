# Google Sheet RFM pipeline

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/gsheet-rfm-pipeline/gsheet-rfm-pipeline){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/gsheet-rfm-pipeline/gsheet-rfm-pipeline/gsheet-rfm-pipeline.ipynb){ .md-button .md-button--secondary }

On this page we will show you how to deploy a pipeline that:

- Retrieves data from a google sheet
- Performs a small RFM analysis on it 
- Writes the top customers back to the google sheet

For this example we will use an opensourse customer data dataset from Kaggle that can be found [here](https://kaggle.com/mrmining/online-retail).

The resulting pipeline in UbiOps will look like this:

![pipeline](https://storage.googleapis.com/ubiops/data/Integration%20with%20other%20tools/gsheet-rfm-pipeline/pictures/pipeline.png)

You can also [download this page](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/gsheet-rfm-pipeline/gsheet-rfm-pipeline){:target="_blank"} as a notebook.

## Preparing your Google environment

In order to run this notebook you will have to prepare a small set-up in your own google environment. Just follow along with the following steps:

1. First, create a google sheet, name it `OnlineRetail` and paste the data from [the OnlineRetail dataset](https://kaggle.com/mrmining/online-retail) in the sheet.

2. Head to [Google Developers Console](https://console.developers.google.com/) and create a new project (or select the one you have.)

3. You will be redirected to the Project Dashboard, there click on "Enable Apis and services", search for "Sheets API".

4. In the API screen click on "ENABLE" to enable this API

![enable_api](https://storage.googleapis.com/ubiops/data/Integration%20with%20other%20tools/gsheet-rfm-pipeline/pictures/api_enable.png)

5. Similarly enable the "Drive API".

Now that we have the base set up, we still need to create a service account to use and give it access to the OnlineRetail data sheet.

6. Go to "Credentials" tab and choose "Create Credentials > Service Account".

7. Give the Service account a name and a description

8. Set the service account permissions as "Compute Engine Service Agent", skip the third step and click create: 

<img src="https://storage.googleapis.com/ubiops/data/Integration%20with%20other%20tools/gsheet-rfm-pipeline/pictures/new_service_account.PNG" width="700">

<img src="https://storage.googleapis.com/ubiops/data/Integration%20with%20other%20tools/gsheet-rfm-pipeline/pictures/permissions.png" width="700">

9. Now navigate to the newly created service account and go to the "Keys" tab. Click "Add Key > Create new Key".

<img src="https://storage.googleapis.com/ubiops/data/Integration%20with%20other%20tools/gsheet-rfm-pipeline/pictures/new_key.png" width="700">

10. Set the type to JSON and click create. This will prompt a download of a json file which contains the necessary
private key for account authorization. Store it in the same folder as this notebook.

<img src="https://storage.googleapis.com/ubiops/data/Integration%20with%20other%20tools/gsheet-rfm-pipeline/pictures/json_key.png" width="700">


Pfew! Okay we are good to continue, everything should be correctly set up now. 


## Sharing the OnlineRetail sheet with the service account

Lastly, we need to make sure the service account actually has access to your sheet. To do this, head over to the Google Sheet you made before and clcik
"share". Share the google sheet with the email address of the service account you created in the previous steps.
The service account will need editor rights, as it will perform both read and write actions.

## Establishing a connection with your UbiOps environment
Add your API token and project name. You can also adapt the deployment name and deployment version name or leave the default values. Afterwards we initialize the client library, which establishes the connection with UbiOps.


```python
API_TOKEN = '<YOUR TOKEN WITH PROJECT EDITOR RIGHTS>' # Should be of the form: Token ah23g4572
PROJECT_NAME= '<YOUR PROJECT NAME>'

# Import all necessary libraries
import shutil
import os
import ubiops
```

Now we can open the connection to UbiOps.


```python
client = ubiops.ApiClient(ubiops.Configuration(api_key={'Authorization': API_TOKEN}, 
                                               host='https://api.ubiops.com/v2.1'))
api = ubiops.CoreApi(client)
api.service_status()
```

And let's define some handy variables we willl be needing often. Please also define the name of your credential json here!


```python
# Deployment configurations
GSHEET_COLLECTOR_DEPLOYMENT='gsheet-data-collector'
RFM_DEPLOYMENT='rfm-model'
GSHEET_WRITER_DEPLOYMENT='gsheet-write-results'
DEPLOYMENT_VERSION='v1'
deployments_list = [GSHEET_COLLECTOR_DEPLOYMENT, RFM_DEPLOYMENT, GSHEET_WRITER_DEPLOYMENT]

# Pipeline configurations
PIPELINE_NAME = 'gsheet-pipeline'
PIPELINE_VERSION = 'v1'

# Your Google credential json
json_filename = '<YOUR JSON FILENAME>' # i.e. 'training-project-2736625.json'
```


```python
os.mkdir("gsheet_input_connector")
os.mkdir("gsheet_output_connector")
os.mkdir("rfm-analysis-package")
```

## Creating the deployment.py for the data collector

In the cell below we create the deployment.py for retrieving data from the google sheet we made earlier. The other files we already prepared in the `gsheet_input_connector` directory.


```python
%%writefile gsheet_input_connector/deployment.py
"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
import json
from google.oauth2 import service_account
import pygsheets
from joblib import dump


class Deployment:

    def __init__(self, base_directory, context):

        print('Initialising the connection to the google drive')
        self.gc = None

        SCOPES = ('https://googleapis.com/auth/spreadsheets', 'https://googleapis.com/auth/drive')
        service_account_info = json.loads(os.environ['credentials'])
        my_credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)

        try:
            self.gc = pygsheets.authorize(custom_credentials=my_credentials)
            print('Established succesfull connection')
        except Exception as e:
            print('Connection failed, ', e.__class__, 'occurred.')

    def request(self, data):

        print('Getting the requested file')
        spreadsheet = self.gc.open(data['filename'])
        sheet_data = spreadsheet[0]

        # UbiOps expects JSON serializable output or files, so we pickle the data
        with open('tmp_sheet.joblib', 'wb') as f:
           dump(sheet_data, 'tmp_sheet.joblib')
        
        return {'data': 'tmp_sheet.joblib'}

```


```python
%%writefile gsheet_input_connector/requirements.txt

cachetools==4.2.2
certifi==2020.12.5
chardet==4.0.0
google-api-core==1.28.0
google-api-python-client==2.5.0
google-auth==1.30.0
google-auth-httplib2==0.1.0
google-auth-oauthlib==0.4.4
googleapis-common-protos==1.53.0
httplib2==0.19.1
idna==2.10
joblib==1.0.1
numpy==1.20.3
oauthlib==3.1.0
packaging==20.9
protobuf==3.17.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pygsheets==2.0.5
pyparsing==2.4.7
pytz==2021.1
requests==2.25.1
requests-oauthlib==1.3.0
rsa==4.7.2
six==1.16.0
uritemplate==3.0.1
urllib3==1.26.4

```

## Deploying the data collector

Now that our deployment package is ready we can deploy it to UbiOps. In the following cells we define the deployment and upload the code to UbiOps.


```python
deployment_template = ubiops.DeploymentCreate(
    name=GSHEET_COLLECTOR_DEPLOYMENT,
    description='Collects data from a google sheet.',
    input_type='structured',
    output_type='structured',
    input_fields=[{'name':'filename', 'data_type':'string'}],
    output_fields=[{'name':'data', 'data_type':'file'}]
)

deployment = api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
print(deployment)
```


```python
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-7',
    instance_type='256mb',
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='none' # We don't need request storage
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=GSHEET_COLLECTOR_DEPLOYMENT,
    data=version_template
)
print(version)
```

Now we create the required environment variable and zip and upload the code package.


```python
# Read in the credentials json into a string
with open(json_filename) as json_file:
    cred_json = json_file.read().strip()

# Create the environment variable to contain the credentials
env_var_response = api.deployment_environment_variables_create(
    project_name=PROJECT_NAME,        
    deployment_name=GSHEET_COLLECTOR_DEPLOYMENT,
    data= {
      "name": "credentials",
      "value": cred_json,
      "secret": True
    }
)
print(env_var_response)


# Zip the deployment package
shutil.make_archive('gsheet_input_connector', 'zip', '.', 'gsheet_input_connector')

upload_response1 = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=GSHEET_COLLECTOR_DEPLOYMENT,
    version=DEPLOYMENT_VERSION,
    file='gsheet_input_connector.zip'
)
print(upload_response1)
```

Right now there should be a deployment called `gsheet-data-collector` visible in the WebApp under the deployments tab. It should have one version that is building or available. While that one is building we can continue to create the other deployments we need.

## Creating the RFM deployment

First we have to create the deployment.py we need, which we do in the cell below.


```python
%%writefile rfm-analysis-package/deployment.py
"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import pygsheets
from joblib import load, dump
import pandas as pd


class Deployment:

    def __init__(self, base_directory, context):

        print('Initalizing model')

    def request(self, data):

        print('Loading the data')
        sheet_data = load(data['retail_data'])

        # Transforming it into a Pandas DataFrame
        data_df = sheet_data.get_as_df()

        # RFM analyis
        print('Performing RFM analysis')
        data_df['TotalPrice'] = data_df['Quantity'].astype(int) * data_df['UnitPrice'].astype(float)
        data_df['InvoiceDate'] = pd.to_datetime(data_df['InvoiceDate'])

        rfm= data_df.groupby('CustomerID').agg({'InvoiceDate': lambda date: (date.max() - date.min()).days,
                                                'InvoiceNo': lambda num: len(num),
                                                'TotalPrice': lambda price: price.sum()})

        # Change the name of columns
        rfm.columns=['recency','frequency','monetary']

        # Computing Quantile of RFM values
        rfm['recency'] = rfm['recency'].astype(int)
        rfm['r_quartile'] = pd.qcut(rfm['recency'].rank(method='first'), 4, ['1','2','3','4']).astype(int)
        rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1']).astype(int)
        rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1']).astype(int)

        rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)

        # Filter out Top/Best customers
        print('Filtering out top customers')
        top_customers = rfm[rfm['RFM_Score']=='111'].sort_values('monetary', ascending=False)        

        # UbiOps expects JSON serializable output or files, so we pickle the data
        with open('top_customers.joblib', 'wb') as f:
           dump(top_customers, 'top_customers.joblib')
        
        return {'top_customers': 'top_customers.joblib'}

```


```python
%%writefile rfm-analysis-package/requirements.txt
joblib==1.0.1
numpy==1.20.3
pygsheets==2.0.5
pandas==1.2.4
```

Just like before for the collector, we create the deployment for the rfm analysis.


```python
deployment_template = ubiops.DeploymentCreate(
    name=RFM_DEPLOYMENT,
    description='RFM analysis that filters out the top customers.',
    input_type='structured',
    output_type='structured',
    input_fields=[{'name':'retail_data', 'data_type':'file'}],
    output_fields=[{'name':'top_customers', 'data_type':'file'}]
)

deployment = api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
print(deployment)
```


```python
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-8',
    instance_type='1024mb',
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='none' # We do not need request storage
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=RFM_DEPLOYMENT,
    data=version_template
)
print(version)
```


```python
# Zip the deployment package
shutil.make_archive('rfm-analysis-package', 'zip', '.', 'rfm-analysis-package')

upload_response2 = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=RFM_DEPLOYMENT,
    version=DEPLOYMENT_VERSION,
    file='rfm-analysis-package.zip'
)
print(upload_response2)
```

The RFM model should now also be building in your UbiOps environment. Time to move on to the last deployment we need, the output connector.

## Deploying the output connector

Just like before we will first create a deployment.py, then a deployment and the required environment variables, after which we upload the code to UbiOps.


```python
%%writefile gsheet_output_connector/deployment.py
"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
import json
from google.oauth2 import service_account
import pygsheets
from joblib import load
import pandas


class Deployment:

    def __init__(self, base_directory, context):

        print('Initialising the connection to the google drive')
        self.gc = None

        SCOPES = ('https://googleapis.com/auth/spreadsheets', 'https://googleapis.com/auth/drive')
        service_account_info = json.loads(os.environ['credentials'])
        my_credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)

        try:
            self.gc = pygsheets.authorize(custom_credentials=my_credentials)
            print('Established succesfull connection')
        except Exception as e:
            print('Connection failed, ', e.__class__, 'occurred.')

    def request(self, data):

        print('Loading top customers')
        top_customers = load(data['data'])

        print('Inserting data into the google sheet')
        spreadsheet = self.gc.open(os.environ['filename'])
        sheet_title = os.environ['sheet_title']

        try:
            sh = spreadsheet.worksheet_by_title(sheet_title)
        except:
            print('Worksheet does not exist, adding new sheet')
            spreadsheet.add_worksheet(sheet_title)
            sh = spreadsheet.worksheet_by_title(sheet_title)
        finally:
            sh.set_dataframe(top_customers, 'A1', copy_index = True)
            sh.update_value('A1', 'CustomerID')
            print('Data inserted successfully')     
        

```


```python
%%writefile gsheet_output_connector/requirements.txt

google-api-core==1.28.0
google-api-python-client==2.5.0
google-auth==1.30.0
google-auth-httplib2==0.1.0
google-auth-oauthlib==0.4.4
googleapis-common-protos==1.53.0
joblib==1.0.1
numpy==1.20.3
oauthlib==3.1.0
pygsheets==2.0.5
pandas==1.2.4
```


```python
deployment_template = ubiops.DeploymentCreate(
    name=GSHEET_WRITER_DEPLOYMENT,
    description='Gsheet output connector',
    input_type='structured',
    output_type='structured',
    input_fields=[{'name':'data', 'data_type':'file'}],
    output_fields=[]
)

deployment = api.deployments_create(project_name=PROJECT_NAME, data=deployment_template)
print(deployment)
```


```python
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-7',
    instance_type='1024mb',
    maximum_instances=1,
    minimum_instances=0,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='none' # We don't need request storage
)

version = api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=GSHEET_WRITER_DEPLOYMENT,
    data=version_template
)
print(version)
```


```python
# Create the environment variable to contain the credentials
env_var_response = api.deployment_environment_variables_create(
    project_name=PROJECT_NAME,        
    deployment_name=GSHEET_WRITER_DEPLOYMENT,
    data= {
      'name': 'credentials',
      'value': cred_json,
      'secret': True
    }
)
print(env_var_response)

# Create the environment variable for the filename
env_var_response = api.deployment_environment_variables_create(
    project_name=PROJECT_NAME,        
    deployment_name=GSHEET_WRITER_DEPLOYMENT,
    data= {
      'name': 'filename',
      'value': 'OnlineRetail',
      'secret': False
    }
)
print(env_var_response)

# Create the environment variable for the sheet title
# This is the sheet to which the results will be written
env_var_response = api.deployment_environment_variables_create(
    project_name=PROJECT_NAME,        
    deployment_name=GSHEET_WRITER_DEPLOYMENT,
    data= {
      'name': 'sheet_title',
      'value': 'Top Customers',
      'secret': False
    }
)
print(env_var_response)


# Zip the deployment package
shutil.make_archive('gsheet_output_connector', 'zip', '.', 'gsheet_output_connector')

upload_response3 = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=GSHEET_WRITER_DEPLOYMENT,
    version=DEPLOYMENT_VERSION,
    file='gsheet_output_connector.zip'
)
print(upload_response3)
```

## Waiting for the deployments to finish building

Right now all three deployments are building, and we need to wait until they are available before we proceed. The following while loop checks if they are available.


```python
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=GSHEET_COLLECTOR_DEPLOYMENT,
    version=DEPLOYMENT_VERSION,
    revision_id=upload_response1.revision
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=RFM_DEPLOYMENT,
    version=DEPLOYMENT_VERSION,
    revision_id=upload_response2.revision
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name=GSHEET_WRITER_DEPLOYMENT,
    version=DEPLOYMENT_VERSION,
    revision_id=upload_response3.revision
)
```

## Making the pipeline

Now that we have our three deployments we can connect them together in a pipeline. Our pipeline will first call the data collector, then the RFM analysis and lastly it will write away the results to a separate sheet in the google spreadsheet.


```python
pipeline_template = ubiops.PipelineCreate(
    name=PIPELINE_NAME,
    description='A simple pipeline that performs an RFM analysis on retail data from a google sheet.',
    input_type='structured',
    input_fields=[{'name':'filename', 'data_type':'string'}],
    output_type='structured',
    output_fields=[]
)

api.pipelines_create(project_name=PROJECT_NAME, data=pipeline_template)
```

### Create the pipeline version

Now that we have a pipeline, we can create a version with the actual deployments in there.


```python
pipeline_template = ubiops.PipelineVersionCreate(
    version=PIPELINE_VERSION,
    request_retention_mode='full',
    objects=[
        # input connector
        {
            'name': GSHEET_COLLECTOR_DEPLOYMENT,
            'reference_name': GSHEET_COLLECTOR_DEPLOYMENT,
            'version': DEPLOYMENT_VERSION
        },
        # RFM model
        {
            'name': RFM_DEPLOYMENT,
            'reference_name': RFM_DEPLOYMENT,
            'version': DEPLOYMENT_VERSION
        },
        # output connector
        {
            'name': GSHEET_WRITER_DEPLOYMENT,
            'reference_name': GSHEET_WRITER_DEPLOYMENT,
            'version': DEPLOYMENT_VERSION
        }
    ],
    attachments=[
        # start --> data collector
        {
            'destination_name': GSHEET_COLLECTOR_DEPLOYMENT,
            'sources': [{
                'source_name': 'pipeline_start',
                'mapping': [{
                    "source_field_name": 'filename',
                    'destination_field_name': 'filename'
                }]
            }]
        },
        # Data collector -> RFM model
        {
            'destination_name': RFM_DEPLOYMENT,
            'sources': [{
                'source_name': GSHEET_COLLECTOR_DEPLOYMENT,
                'mapping': [{
                    "source_field_name": 'data',
                    'destination_field_name': 'retail_data'
                }]
            }]
        },
        # RFM model -> output connector
        {
            'destination_name': GSHEET_WRITER_DEPLOYMENT,
            'sources': [{
                'source_name': RFM_DEPLOYMENT,
                'mapping': [{
                    "source_field_name": 'top_customers',
                    'destination_field_name': 'data'
                }]
            }]
        }
    ]
)

api.pipeline_versions_create(project_name=PROJECT_NAME, pipeline_name=PIPELINE_NAME, data=pipeline_template)
```

If all went well you should have a pipeline that looks like this:

![pipeline](https://storage.googleapis.com/ubiops/data/Integration%20with%20other%20tools/gsheet-rfm-pipeline/pictures/pipeline.png)


## Making a request

With our pipeline done, we can send a request to it perform the RFM analysis on our OnlineRetail sheet! Run the cell below to do so.
The RFM analysis is not that fast so the request might take a little while to complete. You can check the logs in the WebApp to see what is going on in the background.


```python
data = {'filename':'OnlineRetail'}
pipeline_result = api.pipeline_version_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=PIPELINE_NAME,
    version=PIPELINE_VERSION,
    data=data
)
print(pipeline_result)
```

**Note**: This notebook runs on Python 3.8 and uses UbiOps CLient Library 3.15.0.

## Exploring further
You can go ahead to the WebApp and take a look in the user interface at what you have just built and explore further.

So there we have it! We have created a pipeline that itneracts with a Google Sheet. You can use this notebook to base your own deployments on. Just adapt the code in the deployment packages and alter the input and output fields as you wish and you should be good to go.

For any questions, feel free to reach out to us via the customer service portal: https://ubiops.atlassian.net/servicedesk/customer/portals

## Disabling the service account

Tip: disable or delete your service account in the google console if you do not plan on using it anymore. You can do so by navigating to the service account and clicking "Disable service account", or "Delete service account". 
