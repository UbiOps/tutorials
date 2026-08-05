# Number multiplication

<div class="videoWrapper">

<iframe src="https://youtube.com/embed/2MRtdv-3bwU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

</div>

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/multiplication/deployment_package){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/ready-deployments/multiplication/deployment_package){ .md-button .md-button--secondary }

To illustrate the basic working of the `deployment.py` required by UbiOps we have created a sample deployment
that multiplies a given number by 2. You can download the deployment package as a zip (ready to be used) [here](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/multiplication/deployment_package).

We have put the `deployment.py` here as well for your reference:

<div id="code">
```python
class Deployment:

    def __init__(self, base_directory, context):

        print("Initialising deployment")

    def request(self, data):

        print("Processing request")
        # Retrieve the user input from the data dictionary
        user_input = data['number']

        # Process the input
        number_multiplied = user_input * 2

        # Return the output
        return {'number_multiplied': number_multiplied}
```
</div>

In the `__init__` method a deployment is initialized. It can for example be used for loading modules that
have to be kept in memory or setting up connections. In this example we do not need that. In the `request`
method we process the given input. We first retrieve the user_input from the data dictionary and then
multiply it by 2. The result of that multiplication is then returned as output `number_multiplied`. 

!!! info "Importing Python modules"
    Please note that this example does not require any Python modules and therefore has no import statements. If you
    need to import external modules then the import statements should precede `Class Deployment`. Be sure to also add
    the dependencies to the `requirements.txt` as well.


## Running the example in UbiOps

To deploy this example model to your own UbiOps environment you can log in to the WebApp and create a new
deployment in the deployment tab. You will be prompted to fill in certain parameters, you can use the
following:

| Deployment configuration | |
|--------------------|--------------|
| Name | multiply |
| Description | Multiplies input by 2 |
| Input fields: | name = number, datatype = integer |
| Output fields: | name = number_multiplied, datatype = integer |
| Version name | v1 |
| Description | leave blank |
| Environment | Python 3.11 |
| Upload code | [deployment zip](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/multiplication/deployment_package) _do not unzip!_ |
| Request retention | Leave on default settings |

The advanced parameters and labels can be left as they are. They are optional.

After uploading the code and with that creating the deployment version UbiOps will start deploying. Once
you're deployment version is available you can make requests to it. You can use any integer as input.
