# UbiOps Pipeline tutorial

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/pipeline-tutorial/pipeline-tutorial){ .md-button .md-button--primary } [View source code :fontawesome-brands-github:](https://github.com/UbiOps/tutorials/blob/master/pipeline-tutorial/pipeline-tutorial/pipeline-tutorial.ipynb){ .md-button }

This tutorial provides a practical overview of all the pipeline functionalities of UbiOps. 
Extra focus will be placed on how to achieve the different functionalities with the UbiOps Python SDK.  
Many different examples will be given, getting more complex throughout the tutorial. 

The following items will be explained:

- Basic pipeline functionality
- Output splitting/merging
- Subpipelines
- Variables
- [Operators](https://ubiops.com/docs/pipelines/operators/)
    - [Functions](https://ubiops.com/docs/pipelines/operators/#function)
        - Provides ability to manipulate request data fields
    - [Conditionals](https://ubiops.com/docs/pipelines/operators/#conditional-logic)
        - Provides the ability to conditionally choose the next object in the pipeline
    - [Raise error](https://ubiops.com/docs/pipelines/operators/#raise-error)
        - Provides the ability to raise an error somewhere in the pipeline
    - [Create + Collect Subrequests](https://ubiops.com/docs/pipelines/operators/#create-subrequests)
        - Provides the ability to parallelize subrequests over multiple instances of a deployment object.
    - [Count subrequests](https://ubiops.com/docs/pipelines/operators/#count-subrequests)
        - Provides the ability to count the number of subrequests

Do note that this tutorial will not go through a specific use case.
Instead we will focus on explaining the different pipeline functionalities,
using examples to illustrate solely these functionalities.

## Python SDK

In order to deploy a pipeline using the Python SDK, we will mainly need the following 2 functions:  

- [`pipelines_create`](https://ubiops.com/docs/python_client_library/Pipelines#pipelines_create): This function will create a new pipeline in UbiOps.
- [`pipeline_versions_create`](https://ubiops.com/docs/python_client_library/Pipelines#pipeline_versions_create): This function will create a new version of a pipeline in UbiOps.

We will focus mostly on the `pipeline_versions_create` function, as this is the function that specifies the exact pipeline layout.

### `pipelines_create`

This function is used to create a new pipeline in UbiOps.
This function simply defines the overhead of the pipeline, without specifying the exact layout of the pipeline.  
It requires the following parameters (excluding optional parameters):

- `name`: The name of the pipeline.
- `{input, output}_type`: The input and output type of the pipeline, namely plain or structured.
- `{input, output}_fields`: The fields of the input and output type (when structured).
  
The full documentation can be found on the UbiOps documentation page [here](https://ubiops.com/docs/python_client_library/Pipelines#pipelines_create)

### `pipelines_versions_create`

This function is used to create a new version of a pipeline in UbiOps. This function is the main focus of this tutorial, as it specifies the exact layout of the pipeline.  
It requires the following parameters (excluding optional parameters):

- `project_name`: The name of the project in which the pipeline will be deployed.
- `pipeline_name`: The name of the pipeline.
- `data`: A `PipelineVersionCreate` object, which specifies the exact layout and details of the pipeline.

How we exactly specify the layout of the pipeline with the `PipelineVersionCreate` object will be explained in the following sections.  
The full documentation can be found on the UbiOps documentation page [here](https://ubiops.com/docs/python_client_library/Pipelines#pipeline_versions_create)

#### `PipelineVersionCreate`

Most of the magic happens in the `PipelineVersionCreate` object.
The properties of the `PipelineVersionCreate` class can be found [here](https://ubiops.com/docs/python_client_library/models/PipelineVersionCreate),
with an explanation given [here](https://ubiops.com/docs/python_client_library/Pipelines/#optional-parameters_1).  
This object specifies the exact layout of the pipeline, among other things.  
The most interesting properties of this objects are the `objects` and `attachments` properties. As the name suggests, the `objects` property specifies the different objects in the pipeline, while the `attachments` property specifies the connections between these objects.  


## Setup

Let's set up our environment to start this tutorial.
First, we need to install the UbiOps Python SDK.


```python
!pip install ubiops
```

Next, we need to set up our connection to UbiOps.


```python
import ubiops

API_TOKEN = "<INSERT YOUR TOKEN HERE>" # Make sure this is in the format "Token token-code"
PROJECT_NAME = "<INSERT PROJECT NAME>"

configuration = ubiops.Configuration()
configuration.api_key['Authorization'] = API_TOKEN
configuration.host = 'https://api.ubiops.com/v2.1'

client = ubiops.ApiClient(configuration)
api = ubiops.CoreApi(client)
api.service_status()
```

Now that we have set up our connection to UbiOps, we can get started!

## Basic pipeline functionality

The power of UbiOps pipelines lies in the ability to chain different objects together. Objects can be of different types, such as deployments and operators.  
In this section, we will show how to simply chain deployments together, the most basic functionality, but also most important functionality of UbiOps pipelines.  
In order to showcase this, we will create a simple pipeline that takes an input and does a mathematical operation on it.  


An image of the pipeline we're building in this section is as follows:  

![pipeline](https://storage.googleapis.com/ubiops/tutorial-helper-files/pipeline-tutorial/Basic-Pipeline-Image.png)

This pipeline will consist of 2 deployments, one that multiplies the input by 2 and one that adds 5 to the input.

### Deployment code

For the deployments, we will create 2 different deployments:

- `multiply_2`: This deployment will multiply the input by 2.
- `add_5`: This deployment will add 5 to the input.

These deployments are purely for illustrative purposes, to show how to chain deployments together.

Let's create the deployments by making the corresponding directories and creating the `deployment.py` files.


```python
import os

multiply_2_deployment_name = "multiply-2"
add_5_deployment_name = "add-5"

deployment_directory = "new_deployment"
multiply_2_deployment_path = f"{deployment_directory}/{multiply_2_deployment_name}"
add_5_deployment_path = f"{deployment_directory}/{add_5_deployment_name}"

# Create the directories
os.makedirs(multiply_2_deployment_path, exist_ok=True)
os.makedirs(add_5_deployment_path, exist_ok=True)
```


```python
%%writefile {multiply_2_deployment_path}/deployment.py

class Deployment:
    def __init__(self, base_directory, context):
        pass

    def request(self, data):
        return {
            "output": data["input"] * 2
        }
```


```python
%%writefile {add_5_deployment_path}/deployment.py

class Deployment:
    def __init__(self, base_directory, context):
        pass

    def request(self, data):
        return {
            "output": data["input"] + 5
        }
```

We have now created the deployments. Let's upload these deployments to UbiOps and incorporate them into a pipeline.

### Upload Deployment

To upload the deployment to UbiOps, we need to create the deployment and create a corresponding deployment version.

#### Create Deployment


```python
# Standard input and output variables for the deployments
from ubiops import InputOutputFieldBase

input_variable = InputOutputFieldBase(
    name="input",
    data_type="int"
)
input_variable = [input_variable]

output_variable = InputOutputFieldBase(
    name="output",
    data_type="int"
)
output_variable = [output_variable]
```


```python
deployment_template_multiply = ubiops.DeploymentCreate(
    name=multiply_2_deployment_name,
    description="Deployment that multiplies input by 2",
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

deployment_template_add = ubiops.DeploymentCreate(
    name=add_5_deployment_name,
    description="Deployment that adds 5 to input",
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.deployments_create(PROJECT_NAME, deployment_template_multiply)
api.deployments_create(PROJECT_NAME, deployment_template_add)
```

We have now specified the deployments. Let's create the deployment versions.

#### Create Deployment versions


```python
deployment_version_template = ubiops.DeploymentVersionCreate(
    version="v1",
    environment='python3-11',
    instance_type='512mb',
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=300,  # = 30 minutes
    request_retention_mode='full'
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=multiply_2_deployment_name,
    data=deployment_version_template
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=add_5_deployment_name,
    data=deployment_version_template
)
```

We have now created the deployments and deployment versions. It's now time to upload our deploymnet code to our newly created versions!

#### Zip and upload deployment code


```python
import shutil

add_5_zip_file = shutil.make_archive(add_5_deployment_path, 'zip', add_5_deployment_path)
api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=add_5_deployment_name,
    version="v1",
    file=add_5_zip_file
)

multiply_2_zip_file = shutil.make_archive(multiply_2_deployment_path, "zip", multiply_2_deployment_path)
result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=multiply_2_deployment_name,
    version="v1",
    file=multiply_2_zip_file
)

# # Remove the zip file
os.remove(add_5_zip_file)
os.remove(multiply_2_zip_file)

ubiops.utils.wait_for_deployment_version(client, PROJECT_NAME, add_5_deployment_name, "v1")
ubiops.utils.wait_for_deployment_version(client, PROJECT_NAME, multiply_2_deployment_name, "v1")
```

Our deployments are now ready to be used in a pipeline. Let's create a pipeline that chains these deployments together.

### Create Pipeline

It's time to create the pipeline that chains the deployments together.  
Let's start by creating our pipeline and then our pipeline version. 
Just like with deployments, we need to specify the input and output fields of the pipeline. 
We will use the same input and output variables as the deployments.


```python
basic_pipeline_name = "pipeline-basic-new"

pipeline_template = ubiops.PipelineCreate(
    name=basic_pipeline_name,
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```


```python
pipeline_multiply_name = multiply_2_deployment_name
pipeline_add_name = add_5_deployment_name

pipeline_add_object = {
    "name": pipeline_add_name,
    "reference_type": "deployment",
    "reference_name": add_5_deployment_name,
}

pipeline_multiply_object = {
    "name": pipeline_multiply_name,
    "reference_type": "deployment",
    "reference_name": multiply_2_deployment_name,
}

objects = [
    pipeline_multiply_object,
    pipeline_add_object
]

attachments = [
    # Start -> Multiply Deployment
    {
        "destination_name": pipeline_multiply_name,
        "sources": [
            {
                "source_name": "pipeline_start",
                "mapping": [
                    {
                        "source_field_name": "input",
                        "destination_field_name": "input"
                    }
                ]
            }
        ]
    },
    # Multiply Deployment -> Add Deployment
    {
        "destination_name": pipeline_add_name,
        "sources": [
            {
                "source_name": pipeline_multiply_name,
                "mapping": [
                    {
                        "source_field_name": "output",
                        "destination_field_name": "input"
                    }
                ]
            }
        ]
    },
    # Add Deployment -> End
    {
        "destination_name": "pipeline_end",
        "sources": [
            {
                "source_name": pipeline_add_name,
                "mapping": [
                    {
                        "source_field_name": "output",
                        "destination_field_name": "output"
                    }
                ]
            }
        ]
    }
]

pipeline_version_template = ubiops.PipelineVersionCreate(
    version="v1",
    request_retention_mode='full',
    objects=objects,
    attachments=attachments
)
# Insert or otherwise update pipeline version
api.pipeline_versions_create(
    project_name=PROJECT_NAME,
    pipeline_name="pipeline-basic-new",
    data=pipeline_version_template
)
```

We have now created our first pipeline version with the UbiOps Client Library!  
We can visually inspect this pipeline in the UbiOps WebApp.  
Let's send a request to this pipeline to see if it works as expected.

#### Send pipeline request


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=basic_pipeline_name,
    data={"input": 5}
)
```

As we can see, we've got a successful response from the pipeline!

## Output splitting/merging

In this section, we'll showcase how to split and merge outputs of deployments inside a pipeline.  
The pipeline layout is depicted in the following image:

![pipeline](https://storage.googleapis.com/ubiops/tutorial-helper-files/pipeline-tutorial/Split-Merge-Pipeline-Image.png)

As we can see in the image above,
we're gonna split the output of one deployment into two separate deployments,
and then merge the outputs of these two deployments into one deployment.


### Deployment code

We will use two new deployments to showcase this functionality,
namely a deployment that takes in a single input and returns two outputs,
and a deployment that takes in two inputs and returns a single output.  
Let's create the deployment code for these deployments.


```python
create_2_deployment_name = "create-2-outputs"
merge_2_deployment_name = "sum-2-outputs"

create_2_output_path = f"{deployment_directory}/{create_2_deployment_name}"
merge_2_inputs_path = f"{deployment_directory}/{merge_2_deployment_name}"

os.makedirs(create_2_output_path, exist_ok=True)
os.makedirs(merge_2_inputs_path, exist_ok=True)
```


```python
%%writefile {create_2_output_path}/deployment.py

class Deployment:
    def __init__(self, base_directory, context):
        pass

    def request(self, data):
        return {
            "output_1": data["input"] * 2,
            "output_2": data["input"] * 3
        }

```


```python
%%writefile {merge_2_inputs_path}/deployment.py

class Deployment:
    def __init__(self, base_directory, context):
        pass

    def request(self, data):
        return {
            "output": data["input_1"] + data["input_2"]
        }

```

### Upload Deployment

Once again, it's time to create a deployment and deployment version for the new deployments and upload the deployment code.

#### Create Deployment

Let's specify our deployment input/output variables once again


```python
output_variables_create_2 = [
    InputOutputFieldBase
    (
        name="output_1",
        data_type="int"
    ),
    InputOutputFieldBase(
        name="output_2",
        data_type="int"
    )
]

input_variables_merge_2 = [
    InputOutputFieldBase(
        name="input_1",
        data_type="int"
    ),
    InputOutputFieldBase(
        name="input_2",
        data_type="int"
    )
]

```

Now, we'll create our deployments on UbiOps


```python
deployment_template_create_2 = ubiops.DeploymentCreate(
    name=create_2_deployment_name,
    description="Deployment that creates 2 outputs",
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variables_create_2
)
api.deployments_create(PROJECT_NAME, deployment_template_create_2)
```


```python
deployment_template_merge_2 = ubiops.DeploymentCreate(
    name=merge_2_deployment_name,
    description="Deployment that sums 2 inputs",
    input_type="structured",
    input_fields=input_variables_merge_2,
    output_type="structured",
    output_fields=output_variable
)

api.deployments_create(PROJECT_NAME, deployment_template_merge_2)
```

#### Create Deployment versions

Now that we have created the deployments, we can create the deployment versions.


```python
deployment_version_template = ubiops.DeploymentVersionCreate(
    version="v1",
    environment='python3-11',
    instance_type='512mb',
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=300,  # = 30 minutes
    request_retention_mode='full'
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=create_2_deployment_name,
    data=deployment_version_template
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=merge_2_deployment_name,
    data=deployment_version_template
)
```

#### Zip and upload deployment code

Let's create the zip files and upload the deployment code to UbiOps.


```python
create_2_zip_file = shutil.make_archive(create_2_output_path, 'zip', create_2_output_path)
api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=create_2_deployment_name,
    version="v1",
    file=create_2_zip_file
)
os.remove(create_2_zip_file)
```


```python
merge_2_zip_file = shutil.make_archive(merge_2_inputs_path, "zip", merge_2_inputs_path)
result = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=merge_2_deployment_name,
    version="v1",
    file=merge_2_zip_file
)
os.remove(merge_2_zip_file)
```


```python
ubiops.utils.wait_for_deployment_version(client, PROJECT_NAME, create_2_deployment_name, "v1")
ubiops.utils.wait_for_deployment_version(client, PROJECT_NAME, merge_2_deployment_name, "v1")
```

### Create pipeline

The following code block will create the pipeline as depicted in the previously showcased image.
In this example, we will add one specific deployment twice to the pipeline to process each output separately. 
Note that these deployments will be seperated inside the pipeline by different reference names.


```python
pipeline_split_merge_name = "pipeline-split-merge"

# Create pipeline
pipeline_template = ubiops.PipelineCreate(
    name=pipeline_split_merge_name,
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```


```python
create_2_pipeline_name = create_2_deployment_name
add_5_pipeline_name_1 = f"{add_5_deployment_name}-1"
add_5_pipeline_name_2 = f"{add_5_deployment_name}-2"
merge_2_pipeline_name = merge_2_deployment_name

objects = [
    {
        "name": create_2_pipeline_name,
        "reference_type": "deployment",
        "reference_name": create_2_deployment_name
    },
    {
        "name": add_5_pipeline_name_1,
        "reference_type": "deployment",
        "reference_name": add_5_deployment_name
    },
    {
        "name": add_5_pipeline_name_2,
        "reference_type": "deployment",
        "reference_name": add_5_deployment_name
    },
    {
        "name": merge_2_pipeline_name,
        "reference_type": "deployment",
        "reference_name": merge_2_deployment_name
    }
]

attachments = [
    {
        "destination_name": create_2_pipeline_name,
        "sources": [
            {
                "source_name": "pipeline_start",
                "mapping": [
                    {
                        "source_field_name": "input",
                        "destination_field_name": "input"
                    }
                ]
            }
        ]
    },
    {
        "destination_name": add_5_pipeline_name_1,
        "sources": [
            {
                "source_name": create_2_pipeline_name,
                "mapping": [
                    {
                        "source_field_name": output_variables_create_2[0].name,
                        "destination_field_name": "input"
                    }
                ]
            }
        ]
    },
    {
        "destination_name": add_5_pipeline_name_2,
        "sources": [
            {
                "source_name": create_2_pipeline_name,
                "mapping": [
                    {
                        "source_field_name": output_variables_create_2[1].name,
                        "destination_field_name": "input"
                    }
                ]
            }
        ]
    },
    {
        "destination_name": merge_2_pipeline_name,
        "sources": [
            {
                "source_name": add_5_pipeline_name_1,
                "mapping": [
                    {
                        "source_field_name": "output",
                        "destination_field_name": input_variables_merge_2[0].name
                    }
                ]
            },
            {
                "source_name": add_5_pipeline_name_2,
                "mapping": [
                    {
                        "source_field_name": "output",
                        "destination_field_name": input_variables_merge_2[1].name
                    }
                ]
            }
        ]
    },
    {
        "destination_name": "pipeline_end",
        "sources": [
            {
                "source_name": merge_2_pipeline_name,
                "mapping": [
                    {
                        "source_field_name": "output",
                        "destination_field_name": "output"
                    }
                ]
            }
        ]
    }
]

pipeline_version_template = ubiops.PipelineVersionCreate(
    version="v1",
    request_retention_mode='full',
    objects=objects,
    attachments=attachments
)
# Insert or otherwise update pipeline version
api.pipeline_versions_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_split_merge_name,
    data=pipeline_version_template
)
```

We can now send a request to this pipeline!

#### Send pipeline request


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_split_merge_name,
    data={"input": 5}
)
```

## Subpipelines

Now we're going to showcase how to use a pipeline inside a pipeline.  
We'll be cascading the two previously created pipelines into a single pipeline.  
The pipeline layout is depicted in the following image:

![Subpipelines-image](https://storage.googleapis.com/ubiops/tutorial-helper-files/pipeline-tutorial/Pipeline-in-pipeline.png)

For this pipeline, we won't be needing to upload any new deployments. We can therefore immediately start creating our pipeline and specifying our pipeline version!

### Create Pipeline

Let's create our pipeline and corresponding version:


```python
subpipeline_name = "subpipeline-pipeline"

# Create pipeline
pipeline_template = ubiops.PipelineCreate(
    name=subpipeline_name,
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```


```python
basic_pipeline_pipeline_name = basic_pipeline_name
split_merge_pipeline_pipeline_name = pipeline_split_merge_name

objects = [
    {
        "name": basic_pipeline_pipeline_name,
        "reference_type": "pipeline",
        "reference_name": basic_pipeline_pipeline_name
    },
    {
        "name": split_merge_pipeline_pipeline_name,
        "reference_type": "pipeline",
        "reference_name": split_merge_pipeline_pipeline_name
    }
]

attachments = [
    {
        "destination_name": basic_pipeline_pipeline_name,
        "sources": [
            {
                "source_name": "pipeline_start",
                "mapping": [
                    {
                        "source_field_name": "input",
                        "destination_field_name": "input"
                    }
                ]
            }
        ]
    },
    {
        "destination_name": split_merge_pipeline_pipeline_name,
        "sources": [
            {
                "source_name": basic_pipeline_pipeline_name,
                "mapping": [
                    {
                        "source_field_name": "output",
                        "destination_field_name": "input"
                    }
                ]
            }
        ]
    },
    {
        "destination_name": "pipeline_end",
        "sources": [
            {
                "source_name": split_merge_pipeline_pipeline_name,
                "mapping": [
                    {
                        "source_field_name": "output",
                        "destination_field_name": "output"
                    }
                ]
            }
        ]
    }
]

pipeline_version_template = ubiops.PipelineVersionCreate(
    version="v1",
    request_retention_mode='full',
    objects=objects,
    attachments=attachments
)
# Insert or otherwise update pipeline version
api.pipeline_versions_create(
    project_name=PROJECT_NAME,
    pipeline_name=subpipeline_name,
    data=pipeline_version_template
)
```

We can now send a request to this pipeline!

#### Send pipeline request


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=subpipeline_name,
    data={"input": 5}
)
```

## Variables

In this section, we will showcase pipeline variables.
Pipeline variables are fixed variables that can be used in the pipeline.  
The pipeline layout is depicted in the following image:

![pipeline_variable_image](https://storage.googleapis.com/ubiops/tutorial-helper-files/pipeline-tutorial/Pipeline_Variable.png)

In this pipeline, we'll be summing the input with the value of a pipeline variable.  
Since we already have created the `sum-2-outputs` deployment in a previous stage, we won't be needing to upload any new deployments in this section. We can therefore immediately start with our pipeline code!

### Pipeline code


```python
pipeline_variable_name = "pipeline-variable"

# Create pipeline
pipeline_template = ubiops.PipelineCreate(
    name=pipeline_variable_name,
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```


```python
# Variable Object
variable_object = {
    "name": "variable",
    "reference_name": "pipeline-variable",
    "reference_type": "operator",
    "configuration": {
        "input_fields": [],
        "output_fields": [{"name": "variable", "data_type": "int"}],
        "output_values": [{"name": "variable", "value": 1}]
    }
}

objects = [
    variable_object,
    {
        "name": merge_2_pipeline_name,
        "reference_type": "deployment",
        "reference_name": merge_2_deployment_name
    }
]

attachments = [
    # Pipeline Start + Variable -> Sum Inputs Deployment
    {
        "destination_name": merge_2_pipeline_name,
        "sources": [
            # Pipeline Start
            {
                "source_name": "pipeline_start",
                "mapping": [{
                    "source_field_name": "input",
                    "destination_field_name": input_variables_merge_2[0].name
                }]
            },
            # Variable
            {
                "source_name": variable_object["name"],
                "mapping": [{
                    "source_field_name": variable_object["configuration"]["output_fields"][0]["name"],
                    "destination_field_name": input_variables_merge_2[1].name
                }]

            }]
    },
    # Sum Inputs Deployment -> Pipeline End
    {
        "destination_name": "pipeline_end",
        "sources": [{
            "source_name": merge_2_pipeline_name,
            "mapping": [{
                "source_field_name": "output",
                "destination_field_name": "output"
            }]
        }]
    }
]

pipeline_version_template = ubiops.PipelineVersionCreate(
    version="v1",
    request_retention_mode='full',
    objects=objects,
    attachments=attachments
)
# Insert or otherwise update pipeline version
api.pipeline_versions_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_variable_name,
    data=pipeline_version_template
)
```

We can now send a request to this pipeline!

#### Send pipeline request


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_variable_name,
    data={"input": 5}
)
```

## Operators


In this section, we will discuss the different operators that you can use with the UbiOps pipeline functionality.  
The following operators will be discussed:

- [Functions](https://ubiops.com/docs/pipelines/operators/#function)  
    - Provides the ability to manipulate request data fields.
- [Conditionals](https://ubiops.com/docs/pipelines/operators/#conditional-logic)  
    - Provides the ability to conditionally choose the next object in the pipeline.
- [Raise error](https://ubiops.com/docs/pipelines/operators/#raise-error)  
    - Provides the ability to raise an error somewhere in the pipeline.
- [Create + Collect Subrequests](https://ubiops.com/docs/pipelines/operators/#create-subrequests)  
    - Provides the ability to parallelize subrequests over multiple instances of a deployment object.
- [Count subrequests](https://ubiops.com/docs/pipelines/operators/#count-subrequests)  
    - Provides the ability to count the number of subrequests.

### Functions

Functions provide the ability to manipulate request data fields. 
In this example, we will create a function that multiplies the input by 5.  
The layout of the pipeline is depicted in the following image:

![pipeline_function_image](https://storage.googleapis.com/ubiops/tutorial-helper-files/pipeline-tutorial/Pipeline_Function.png)

In this pipeline, we'll be multiplying the pipeline input by 5 with a pipeline function.  
As we can see in the image, we only use a function in the pipeline. This can be specified in the pipeline code, so we can start creating our pipeline code immediately!

### Pipeline Code


```python
pipeline_function_name = "pipeline-function"

# Create pipeline
pipeline_template = ubiops.PipelineCreate(
    name=pipeline_function_name,
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```


```python
# Use same variable names as pipeline
function_name = "operator-function"

# Create the function object
function_object = {
    "name": function_name,
    "reference_name": "function",
    "reference_type": "operator",
    "configuration": {
        "expression": f"input * 5",
        "input_fields": [
            {
                "name": "input",
                "data_type": "int"
            }
        ],
        "output_fields": [
            {
                "name": "output",
                "data_type": "int"
            }
        ]
    }
}

objects = [function_object]
# Specify the attachments
attachments = [
    # Pipeline start -> Function
    {
        'destination_name': function_name,
        'sources': [{
            'source_name': 'pipeline_start',
            'mapping': [{
                "source_field_name": "input",
                "destination_field_name": "input"
            }]
        }]
    },
    # Function -> Pipeline end
    {
        'destination_name': 'pipeline_end',
        'sources': [{
            'source_name': function_name,
            'mapping': [{
                "source_field_name": "output",
                "destination_field_name": "output"
            }]
        }]
    }
]

pipeline_version_template = ubiops.PipelineVersionCreate(
    version="v1",
    request_retention_mode='full',
    objects=objects,
    attachments=attachments
)
# Insert or otherwise update pipeline version
api.pipeline_versions_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_function_name,
    data=pipeline_version_template
)
```

We can now send a request to this pipeline!

#### Send pipeline request


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_function_name,
    data={"input": 5}
)
```

### Conditionals

The Conditional Logic operator provides the ability to conditionally choose the next object in the pipeline.  
In this example, we will create a conditional that checks if the input is greater than or equal to 10. 
Depending on the outcome, either one of two variables will be used in the next deployment.  
The layout of the pipeline is depicted in the following image:

![pipeline_condition_image](https://storage.googleapis.com/ubiops/tutorial-helper-files/pipeline-tutorial/Pipeline_Conditional.png)

Since we have already deployed the `add-5` deployment, we can continue immediately to the pipeline code!

### Pipeline code


```python
pipeline_conditional_name = "pipeline-conditional"

# Create pipeline
pipeline_template = ubiops.PipelineCreate(
    name=pipeline_conditional_name,
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```


```python
variable1_name = "variable1"
variable2_name = "variable2"
variable_objects = [{
    "name": name,
    "reference_name": "pipeline-variable",
    "reference_type": "operator",
    "configuration": {
        "input_fields": [],
        "output_fields": [{"name": name, "data_type": "int"}],
        "output_values": [{"name": name, "value": value}]
    }
} for name, value in zip([variable1_name, variable2_name], [10, 20])]

input_condition_name = "input"

if_condition = {
    "name": "if-condition",
    "reference_type": "operator",
    "reference_name": "if-condition",
    "configuration": {
        "expression": f"input < 10",
        "input_fields": [
            {
                "name": "input",
                "data_type": "int"
            }
        ],
        "output_fields": []
    }
}

else_condition = {
    "name": "else-condition",
    "reference_type": "operator",
    "reference_name": "if-condition",
    "configuration": {
        "expression": f"input >= 10",
        "input_fields": [
            {
                "name": "input",
                "data_type": "int"
            }
        ],
        "output_fields": []
    }
}

objects = [if_condition, else_condition, pipeline_add_object] + variable_objects

attachments = [
    # pipeline_start -> if_condition
    {
        'destination_name': if_condition["name"],
        'sources': [{
            'source_name': 'pipeline_start',
            'mapping': [{
                "source_field_name": "input",
                "destination_field_name": "input"
            }]
        }]
    },
    # pipeline_start -> else_condition
    {
        'destination_name': else_condition["name"],
        'sources': [{
            'source_name': 'pipeline_start',
            'mapping': [{
                "source_field_name": "input",
                "destination_field_name": "input"
            }]
        }]
    },
    # if_condition + variable1 -> add_5_deployment
    {
        'destination_name': pipeline_add_name,
        'sources': [
            {
                'source_name': if_condition["name"],
                'mapping': []
            },
            {
                "source_name": variable1_name,
                "mapping": [{
                    "source_field_name": variable1_name,
                    "destination_field_name": "input"
                }]
            }]
    },
    # else_condition + variable2 -> add_5_deployment
    {
        'destination_name': pipeline_add_name,
        'sources': [
            {
                'source_name': else_condition["name"],
                'mapping': []
            },
            {
                "source_name": variable2_name,
                "mapping": [{
                    "source_field_name": variable2_name,
                    "destination_field_name": "input"
                }]
            }]
    },
    # add_5_deployment -> pipeline_end
    {
        'destination_name': 'pipeline_end',
        'sources': [{
            'source_name': pipeline_add_name,
            'mapping': [{
                "source_field_name": "output",
                "destination_field_name": "output"
            }]
        }]
    }
]

# Insert or otherwise update pipeline version
pipeline_version_template = ubiops.PipelineVersionCreate(
    version="v1",
    request_retention_mode='full',
    objects=objects,
    attachments=attachments
)
# Insert or otherwise update pipeline version
api.pipeline_versions_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_conditional_name,
    data=pipeline_version_template
)
```

We can now send a request to this pipeline!

#### Send pipeline request


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_conditional_name,
    data={"input": 5}
)
```

### Raise error

The `Raise Error` operator provides the ability to raise an error somewhere in the pipeline.  
In this example, we'll build a pipeline that throws an error if the input is greater than or equal to 10.  
The layout of the pipeline is depicted in the following image:

![pipeline_raise_error_image](https://storage.googleapis.com/ubiops/tutorial-helper-files/pipeline-tutorial/Pipeline_Raise_Error.png)

Let's dive straight into the pipeline code

### Pipeline code


```python
pipeline_raise_name = "pipeline-raise"

# Create pipeline
pipeline_template = ubiops.PipelineCreate(
    name=pipeline_raise_name,
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```


```python
if_condition = {
    "name": "if-condition",
    "reference_type": "operator",
    "reference_name": "if-condition",
    "configuration": {
        "expression": f"input >= 10",
        "input_fields": [
            {
                "name": "input",
                "data_type": "int"
            }
        ],
        "output_fields": []
    }
}

raise_error_object = {
    "name": "operator-raise-error",
    "reference_name": "raise-error",
    "reference_type": "operator",
    "configuration": {
        "error_message": "My error message!",
        "input_fields": [],
        "output_fields": []
    }
}

objects = [
    if_condition,
    raise_error_object
]

attachments = [
    # pipeline_start -> if_condition
    {
        'destination_name': if_condition["name"],
        'sources': [{
            'source_name': 'pipeline_start',
            'mapping': [{
                "source_field_name": "input",
                "destination_field_name": "input"
            }]
        }]
    },

    # pipeline_start -> pipeline_end
    {
        'destination_name': 'pipeline_end',
        'sources': [{
            'source_name': "pipeline_start",
            'mapping': [{
                "source_field_name": "input",
                "destination_field_name": "output"
            }]
        }]
    },

    # if_condition -> raise_error
    {
        'destination_name': raise_error_object["name"],
        'sources': [{
            'source_name': if_condition["name"],
            'mapping': []
        }]
    }
]

# Insert or otherwise update pipeline version
pipeline_version_template = ubiops.PipelineVersionCreate(
    version="v1",
    request_retention_mode='full',
    objects=objects,
    attachments=attachments
)
# Insert or otherwise update pipeline version
api.pipeline_versions_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_raise_name,
    data=pipeline_version_template
)
```

We can now send a request to this pipeline!

#### Send pipeline request

##### Error trigger


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_raise_name,
    data={"input": 15}
)
```

##### Success


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_raise_name,
    data={"input": 5}
)
```

### Create + Collect subrequests

In this section, we will focus on the Create and Collect subrequests operators.
These operators take care of the following:  

- `Create Subrequests`: The Create Subrequests operator (`one-to-many`) provides the ability to parallelize subrequests over multiple instances of a deployment object.
For more information, read the corresponding [documentation page](https://ubiops.com/docs/pipelines/operators/#create-subrequests).
- `Collect Subrequests`: The Collect Subrequests operator (`many-to-one`) provides the ability to wait for all parallelized requests to be finished and sent all subrequests as a single list to the next object. Which means that it stops parallelization created by the Create Subrequests operator.
For more information, read the corresponding [documentation page](https://ubiops.com/docs/pipelines/operators/#collect-subrequests).  

The layout of the pipeline is depicted in the following image:  

![pipeline_create_collect](https://storage.googleapis.com/ubiops/tutorial-helper-files/pipeline-tutorial/Pipeline-Create-Collect.png)

In this pipeline, we will calculate the sum of a range of numbers, where 5 will be added to each number in the range of numbers. 
The addition will be done in parallel with the `Create` and `Collect` subrequests operator.  
For this pipeline, we need to introduce 2 new deployments: `create-subrequests` and `collect-subrequests`. Let's start with uploading these deployments to UbiOps!

### Deployment code

Let's create our deployment code to use with the `Create` and `Collect` Subrequests operators.
Note the use of the `request` and `requests` methods for the create and collect deployment respectively.
For more information on the difference between these 2 methods, refer to the [documentation page](https://ubiops.com/docs/pipelines/operators/#collect-subrequests) for more detailed information.


```python
create_subrequests_deployment_name = "create-subrequests-deployment"
collect_subrequests_deployment_name = "collect-subrequests-deployment"

create_subrequests_deployment_path = f"{deployment_directory}/{create_subrequests_deployment_name}"
collect_subrequests_deployment_path = f"{deployment_directory}/{collect_subrequests_deployment_name}"

os.makedirs(create_subrequests_deployment_path, exist_ok=True)
os.makedirs(collect_subrequests_deployment_path, exist_ok=True)
```


```python
%%writefile {create_subrequests_deployment_path}/deployment.py

class Deployment:
    def __init__(self):
        pass

    def request(self, data):
        return [{"output": x} for x in range(data["input"])]

```


```python
%%writefile {collect_subrequests_deployment_path}/deployment.py

class Deployment:
    def __init__(self):
        pass

    def requests(self, data):
        return {"output": sum([x["input"] for x in data])}

```

#### Upload to UbiOps


```python
deployment_template_multiply = ubiops.DeploymentCreate(
    name=create_subrequests_deployment_name,
    description="Deployment that creates output for subrequests",
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

deployment_template_add = ubiops.DeploymentCreate(
    name=collect_subrequests_deployment_name,
    description="Deployment that sums the outputs of subrequests",
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.deployments_create(PROJECT_NAME, deployment_template_multiply)
api.deployments_create(PROJECT_NAME, deployment_template_add)
```


```python
#### Create Deployment versions
deployment_version_template = ubiops.DeploymentVersionCreate(
    version="v1",
    environment='python3-11',
    instance_type='512mb',
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=300,  # = 30 minutes
    request_retention_mode='full'
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=create_subrequests_deployment_name,
    data=deployment_version_template
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=collect_subrequests_deployment_name,
    data=deployment_version_template
)
```


```python
create_subrequests_zip_file = shutil.make_archive(create_subrequests_deployment_path, 'zip', create_subrequests_deployment_path)
api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=create_subrequests_deployment_name,
    version="v1",
    file=create_subrequests_zip_file
)

collect_subrequests_zip_file = shutil.make_archive(collect_subrequests_deployment_path, 'zip', collect_subrequests_deployment_path)
api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=collect_subrequests_deployment_name,
    version="v1",
    file=collect_subrequests_zip_file
)

# # Remove the zip file
os.remove(create_subrequests_zip_file)
os.remove(collect_subrequests_zip_file)

ubiops.utils.wait_for_deployment_version(client, PROJECT_NAME, create_subrequests_deployment_name, "v1")
ubiops.utils.wait_for_deployment_version(client, PROJECT_NAME, collect_subrequests_deployment_name, "v1")
```

Now that all the necessary deployments have been uploaded to UbiOps, we can start with the pipeline code.

### Pipeline code

Now it's time to create our pipeline and upload it to UbiOps


```python
pipeline_create_subrequests_name = "pipeline-create-collect-subrequests"

# Create pipeline
pipeline_template = ubiops.PipelineCreate(
    name=pipeline_create_subrequests_name,
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```


```python
pipeline_create_sub_deployment_name = create_subrequests_deployment_name
pipeline_collect_sub_name = collect_subrequests_deployment_name

deployment_objects = [
    {
        "name": pipeline_create_sub_deployment_name,
        "reference_type": "deployment",
        "reference_name": create_subrequests_deployment_name,
    },
    {
        "name": pipeline_collect_sub_name,
        "reference_type": "deployment",
        "reference_name": collect_subrequests_deployment_name,
    }
]

one_to_many_object = {
    "name": "operator-one-to-many",
    "reference_name": "one-to-many",
    "reference_type": "operator",
    "configuration": {
        "batch_size": 2,
        "input_fields": [{"name": "input", "data_type": "int"}],
        "output_fields": [{"name": "input", "data_type": "int"}]
    }
}

many_to_one_object = {
    "name": "operator-many-to-one",
    "reference_name": "many-to-one",
    "reference_type": "operator",
    "configuration": {
        "input_fields": [{"name": "output", "data_type": "int"}],
        "output_fields": [{"name": "output", "data_type": "int"}]
    }
}

objects = deployment_objects + [pipeline_add_object, one_to_many_object, many_to_one_object]

attachments = [
    # pipeline_start -> deployment_create
    {
        'destination_name': pipeline_create_sub_deployment_name,
        'sources': [{
            'source_name': 'pipeline_start',
            'mapping': [{
                "source_field_name": "input",
                "destination_field_name": "input"
            }]
        }]
    },
    # deployment_create -> operator-one-to-many
    {
        'destination_name': one_to_many_object["name"],
        'sources': [{
            'source_name': pipeline_create_sub_deployment_name,
            'mapping': [{
                "source_field_name": "output",
                "destination_field_name": "input"
            }]
        }]
    },
    # operator-one-to-many -> deployment_add_5
    {
        'destination_name': pipeline_add_name,
        'sources': [{
            'source_name': one_to_many_object["name"],
            'mapping': [{
                "source_field_name": "input",
                "destination_field_name": "input"
            }]
        }]
    },
    # deployment_add_5 -> operator-many-to-one
    {
        'destination_name': many_to_one_object["name"],
        'sources': [{
            'source_name': pipeline_add_name,
            'mapping': [{
                "source_field_name": "output",
                "destination_field_name": "output"
            }]
        }]
    },
    # operator-many-to-one -> deployment_collect
    {
        'destination_name': pipeline_collect_sub_name,
        'sources': [{
            'source_name': many_to_one_object["name"],
            'mapping': [{
                "source_field_name": "output",
                "destination_field_name": "input"
            }]
        }]
    },
    # deployment_collect -> pipeline_end
    {
        'destination_name': 'pipeline_end',
        'sources': [{
            'source_name': pipeline_collect_sub_name,
            'mapping': [{
                "source_field_name": "output",
                "destination_field_name": "output"
            }]
        }]
    }
]

# Insert or otherwise update pipeline version
pipeline_version_template = ubiops.PipelineVersionCreate(
    version="v1",
    request_retention_mode='full',
    objects=objects,
    attachments=attachments
)
# Insert or otherwise update pipeline version
api.pipeline_versions_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_create_subrequests_name,
    data=pipeline_version_template
)
```

We can now send a request to this pipeline!

#### Send pipeline request


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_create_subrequests_name,
    data={"input": 5}
)
```

### Count subrequests

At last, we will discuss the Count Subrequests operator. 
The Count Subrequests operator provides the ability to count the number of subrequests.
The layout of the pipeline is depicted in the following image:


![pipeline_count_subrequests](https://storage.googleapis.com/ubiops/tutorial-helper-files/pipeline-tutorial/Pipeline_Count.png)

Since we have uploaded the `create-subrequests` deployment in the previous section, we can immediately continue to the pipeline code.

### Pipeline code


```python
pipeline_count_subrequests_name = "pipeline-count-subrequests"

# Create pipeline
pipeline_template = ubiops.PipelineCreate(
    name=pipeline_count_subrequests_name,
    input_type="structured",
    input_fields=input_variable,
    output_type="structured",
    output_fields=output_variable
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```


```python
count_many_object = {
    "name": "operator-count-many",
    "reference_name": "count-many",
    "reference_type": "operator",
    "configuration": {
        "input_fields": [],
        "output_fields": [{"name": "output", "data_type": "int"}]
    }
}

objects = [
    {
        "name": pipeline_create_sub_deployment_name,
        "reference_type": "deployment",
        "reference_name": create_subrequests_deployment_name,
    },
    count_many_object
]


attachments = [
    # pipeline_start -> deployment_create
    {
        'destination_name': pipeline_create_sub_deployment_name,
        'sources': [{
            'source_name': 'pipeline_start',
            'mapping': [{
                "source_field_name": "input",
                "destination_field_name": "input"
            }]
        }]
    },
    # deployment_create -> count_many
    {
        'destination_name': count_many_object["name"],
        'sources': [{
            'source_name': pipeline_create_sub_deployment_name,
            'mapping': []
        }]
    },
    # count_many -> pipeline_end
    {
        'destination_name': 'pipeline_end',
        'sources': [{
            'source_name': count_many_object["name"],
            'mapping': [{
                "source_field_name": "output",
                "destination_field_name": "output"
            }]
        }]
    }
]

# Insert or otherwise update pipeline version
pipeline_version_template = ubiops.PipelineVersionCreate(
    version="v1",
    request_retention_mode='full',
    objects=objects,
    attachments=attachments
)
# Insert or otherwise update pipeline version
api.pipeline_versions_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_count_subrequests_name,
    data=pipeline_version_template
)
```

We can now send a request to this pipeline!

#### Send pipeline request


```python
api.pipeline_requests_create(
    project_name=PROJECT_NAME,
    pipeline_name=pipeline_count_subrequests_name,
    data={"input": 5}
)
```

## Conclusion

In this tutorial, we have shown you how to create different pipelines with different objects in UbiOps. 
Since we are finished now with UbiOps, let's close the client:



```python
client.close()
```

For any questions, feel free to reach out to us via the [customer service portal](https://ubiops.atlassian.net/servicedesk/customer/portals)!
