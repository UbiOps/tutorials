
# Pokemon pipeline

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/pokemon-pipeline/pokemon-pipeline){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/pokemon-pipeline/pokemon-pipeline/pokemon-pipeline.ipynb){ .md-button .md-button--secondary }

Note: This notebook runs on Python 3.9 and uses UbiOps CLient Library 3.15.0.

If you [download](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/pokemon-pipeline/pokemon-pipeline){:target="_blank"} and run this entire notebook after filling in your access token, the pipeline, which takes a list of pokemon as input and outputs a visualization, and all the necessary models will be deployed to your UbiOps environment. You can thus check your environment after running to explore. You can also check the individual steps on this page to see what we did exactly and how you can adapt it to your own use case.

We recommend to run the cells step by step, as some cells can take a few minutes to finish.

## Establishing a connection with your UbiOps environment
Add your API token. Then we will provide a project name, deployment name and deployment version name. Afterwards we initialize the client library. This way we can deploy the two pipelines to your environment.


```python
API_TOKEN = '<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>'
PROJECT_NAME= '<INSERT PROJECT NAME IN YOUR ACCOUNT>'
DEPLOYMENT_VERSION='v1'

import ubiops
import shutil
import os

client = ubiops.ApiClient(ubiops.Configuration(api_key={'Authorization': API_TOKEN}, 
                                               host='https://api.ubiops.com/v2.1'))
api = ubiops.CoreApi(client)
```

## Initialize local repositories 


```python
os.mkdir("pokemon_matcher")
os.mkdir("pokemon_sorter")
os.mkdir("pokemon_vis")
```

## Creating the deployments



### Pokemon matcher
I found a [dataset on Kaggle](https://kaggle.com/abcsds/pokemon) that has statistics for every Pokémon, which we can use to match names to statistics. When dealing with customers instead of Pokémon this would be replaced by for example your CRM.

In this deployment we do the following steps: 
- Read the input Pokémon
- Read our file with all the Pokémon stats
- Select only the Pokémon we got as input
- Export this selection to a CSV file

See the actual code in the following cell.


```python
%%writefile pokemon_matcher/deployment.py
import pandas as pd
import requests
"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""


class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.
        """

        print("Initialising My Deployment")


    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        with plain output, it is a string. In this example, a dictionary with the key: output.
        """

        selected_pokemon = data.get('input_pokemon')

        pokemon_stats = pd.read_csv("https://storage.googleapis.com/ubiops/data/Working%20with%20pipelines/pokemon-pipeline/Pokemon.csv")

        selected_pokemon_stats = pokemon_stats.loc[pokemon_stats['Name'].isin(selected_pokemon)]

        selected_pokemon_stats.to_csv('selected_pokemon_stats.csv')

        return {
            "output_pokemon": 'selected_pokemon_stats.csv'
        }

```


```python
%%writefile pokemon_matcher/requirements.txt

pandas == 1.4.2
```

Now we create a deployment and a deployment version for the package in the cell above. 


```python
# Zip the models
shutil.make_archive('pokemon_matcher', 'zip', '.','pokemon_matcher')

deployment_template = ubiops.DeploymentCreate(
    name='pokemon-matcher',
    description='Match pokemon names to their stats',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'input_pokemon', 'data_type':'array_string'}
    ],
    output_fields=[
        {'name':'output_pokemon', 'data_type':'file'}
    ],
    labels={'demo': 'pokemon-pipeline'}
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)

# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version='v1',
    environment='python3-9',
    instance_type='512mb',
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='none' # We don't need to store the requests for this deployment
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name='pokemon-matcher',
    data=version_template
)

# Upload the zipped deployment package
upload_response1 = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name='pokemon-matcher',
    version='v1',
    file='pokemon_matcher.zip'
)
```

### Pokemon sorter
We need to sort these Pokémon based on the best stats, we can start with the CSV from the Pokémon matcher step.

In this deployment we perform the following steps: 
- Read the input CSV(from the matcher step)
- Sort them based on their stats(higher is better)
- Export this as a CSV file

See the actual code in the following cell.


```python
%%writefile pokemon_sorter/deployment.py

import pandas as pd

"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""


class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.
        """

        print("Initialising My Deployment")

    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        """

        pokemon_stats = pd.read_csv(data.get('input_pokemon'))

        sorted_pokemon_stats = pokemon_stats.sort_values(["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])

        sorted_pokemon_stats.to_csv('sorted_pokemon_stats.csv')

        return {
            "output_pokemon": 'sorted_pokemon_stats.csv'
        }

```


```python
%%writefile pokemon_sorter/requirements.txt

pandas == 1.4.2
```

Now we create a deployment and a deployment version for the package in the cell above. 


```python
# Zip the models
shutil.make_archive('pokemon_sorter', 'zip', '.','pokemon_sorter')

deployment_template = ubiops.DeploymentCreate(
    name='pokemon-sorter',
    description='Sort pokemon based on their stats',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'input_pokemon', 'data_type':'file'}
    ],
    output_fields=[
        {'name':'output_pokemon', 'data_type':'file'}
    ],
    labels={'demo': 'pokemon-pipeline'}
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)

# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version='v1',
    environment='python3-9',
    instance_type='512mb',
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='none' # We don't need to store the requests for this deployment
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name='pokemon-sorter',
    data=version_template
)

# Upload the zipped deployment package
upload_response2 = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name='pokemon-sorter',
    version='v1',
    file='pokemon_sorter.zip'
)
```

### Pokemon visualizer
I found a great [code snippet](https://kaggle.com/wenxuanchen/pokemon-visualization-radar-chart-t-sne) online for creating Pokémon stat visualizations, just like I remember seeing them on my old “Gameboy color”.

It is not really important to understand what’s happening in this visualization code, just that we can visualize the Pokémon we got from our previous step. The visualisation is then outputted as a PDF.

See the actual code in the following cell.


```python
%%writefile pokemon_vis/deployment.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""


class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.
        """

        print("Initialising My Deployment")

    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.
        """


        # In this order,
        # HP, Defense and Sp. Def will show on left; They represent defense abilities
        # Speed, Attack and Sp. Atk will show on right; They represent attack abilities
        # Attack and Defense, Sp. Atk and Sp. Def will show on opposite positions
        use_attributes = ['Speed', 'Sp. Atk', 'Defense', 'HP', 'Sp. Def', 'Attack']

        pokemon = pd.read_csv(data.get("input_pokemon"))
        df_plot = pokemon

        datas = df_plot[use_attributes].values
        ranges = [[2 ** -20, df_plot[attr].max()] for attr in use_attributes]
        colors = select_color(df_plot['Type 1'])  # select colors based on pokemon Type 1

        fig = plt.figure(figsize=(10, 10))
        radar = RaderChart(fig, use_attributes, ranges)
        for data, color, pokemon in zip(datas, colors, pokemon['Name'].tolist()):
            radar.plot(data, color=color, label=pokemon)
            radar.fill(data, alpha=0.1, color=color)
            radar.legend(loc=1, fontsize='small')
        plt.savefig('pokemon.pdf')

        return {
            "output_pokemon": 'pokemon.pdf'
        }


def _scale_data(data, ranges):
    (x1, x2), d = ranges[0], data[0]
    return [(d - y1) / (y2 - y1) * (x2 - x1) + x1 for d, (y1, y2) in zip(data, ranges)]


class RaderChart():
    def __init__(self, fig, variables, ranges, n_ordinate_levels=6):
        angles = np.arange(0, 360, 360. / len(variables))

        axes = [fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, label="axes{}".format(i)) for i in
                range(len(variables))]
        _, text = axes[0].set_thetagrids(angles, labels=variables)

        for txt, angle in zip(text, angles):
            txt.set_rotation(angle - 90)

        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid("off")

        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            grid_label = [""] + [str(int(x)) for x in grid[1:]]
            ax.set_rgrids(grid, labels=grid_label, angle=angles[i])
            ax.set_ylim(*ranges[i])

        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self, *args, **kw):
        self.ax.legend(*args, **kw)


TYPE_LIST = ['Grass', 'Fire', 'Water', 'Bug', 'Normal', 'Poison',
             'Electric', 'Ground', 'Fairy', 'Fighting', 'Psychic',
             'Rock', 'Ghost', 'Ice', 'Dragon', 'Dark', 'Steel', 'Flying']

COLOR_LIST = ['#8ED752', '#F95643', '#53AFFE', '#C3D221', '#BBBDAF', '#AD5CA2',
              '#F8E64E', '#F0CA42', '#F9AEFE', '#A35449', '#FB61B4', '#CDBD72',
              '#7673DA', '#66EBFF', '#8B76FF', '#8E6856', '#C3C1D7', '#75A4F9']

# The colors are copied from this script: https://kaggle.com/ndrewgele/d/abcsds/pokemon/visualizing-pok-mon-stats-with-seaborn
# The colors look reasonable in this map: For example, Green for Grass, Red for Fire, Blue for Water...
COLOR_MAP = dict(zip(TYPE_LIST, COLOR_LIST))


# select display colors according to Pokemon's Type 1
def select_color(types):
    colors = [None] * len(types)
    used_colors = set()
    for i, t in enumerate(types):
        curr = COLOR_MAP[t]
        if curr not in used_colors:
            colors[i] = curr
            used_colors.add(curr)
    unused_colors = set(COLOR_LIST) - used_colors
    for i, c in enumerate(colors):
        if not c:
            try:
                colors[i] = unused_colors.pop()
            except:
                raise Exception('Attempt to visualize too many pokemon. No more colors available.')
    return colors

```


```python
%%writefile pokemon_vis/requirements.txt

pandas == 1.4.2
seaborn == 0.12.1
matplotlib == 3.5.1
```

Now we create a deployment and a deployment version for the package in the cell above. 


```python
# Zip the models
shutil.make_archive('pokemon_vis', 'zip', '.','pokemon_vis')

deployment_template = ubiops.DeploymentCreate(
    name='pokemon-vis',
    description='Visualize the results.',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'input_pokemon', 'data_type':'file'}
    ],
    output_fields=[
        {'name':'output_pokemon', 'data_type':'file'}
    ],
    labels={'demo': 'pokemon-pipeline'}
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)

# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version='v1',
    environment='python3-9',
    instance_type='512mb',
    minimum_instances=0,
    maximum_instances=1,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='none' # We don't need to store the requests for this deployment
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name='pokemon-vis',
    data=version_template
)

# Upload the zipped deployment package
upload_response3 = api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name='pokemon-vis',
    version='v1',
    file='pokemon_vis.zip'
)
```

## Creating the Pokemon pipeline

So right now we have three deployments. We want to tie these blocks together. We can use pipelines for that. Let's create a pipeline that takes the same input as the pokemon_sorter deployment.


```python
pipeline_name = "pokemon-pipeline"
PIPELINE_VERSION = DEPLOYMENT_VERSION

pipeline_template = ubiops.PipelineCreate(
    name=pipeline_name,
    description='',
    input_type='structured',
    input_fields=[
        {'name':'input_pokemon', 'data_type':'array_string'}
    ],
    output_type='structured',
    output_fields=[
        {'name':'output_pokemon', 'data_type':'file'}
    ],
    labels={'demo': 'pokemon-pipeline'}
)

api.pipelines_create(
    project_name=PROJECT_NAME,
    data=pipeline_template
)
```

We have a pipeline, now we just need to make a version and add our objects.

**IMPORTANT**: only run the next cells once your deployments have finished building and are available. Otherwise you will get an error like: "error":"Version is not available: The version is currently in the building stage"

If you get this error, check in the UI if your model is ready and then rerun the cell below.


```python
ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name='pokemon-matcher',
    version='v1',
    revision_id=upload_response1.revision
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name='pokemon-sorter',
    version='v1',
    revision_id=upload_response2.revision
)

ubiops.utils.wait_for_deployment_version(
    client=api.api_client,
    project_name=PROJECT_NAME,
    deployment_name='pokemon-vis',
    version='v1',
    revision_id=upload_response3.revision
)
```


```python
pipeline_template = ubiops.PipelineVersionCreate(
    version=PIPELINE_VERSION,
    request_retention_mode='full',
    objects=[
        # pokemon matcher
        {
            'name': "pokemon-matcher",
            'reference_name': "pokemon-matcher",
            'version': DEPLOYMENT_VERSION
        },
        # pokemon sorter
        {
            'name': "pokemon-sorter",
            'reference_name': "pokemon-sorter",
            'version': DEPLOYMENT_VERSION
        },
        # pokemon visualizer
        {
            'name': "pokemon-vis",
            'reference_name': "pokemon-vis",
            'version': DEPLOYMENT_VERSION
        }
    ],
    attachments=[
        # start -> pokemon-matcher
        {
            'destination_name': "pokemon-matcher",
            'sources': [{
                'source_name': 'pipeline_start',
                'mapping': [{
                    "source_field_name": 'input_pokemon',
                    'destination_field_name': 'input_pokemon'
                }]
            }]
        },
        # pokemon-matcher -> pokemon-sorter
        {
            'destination_name': "pokemon-sorter",
            'sources': [{
                'source_name': "pokemon-matcher",
                'mapping': [{
                    "source_field_name": 'output_pokemon',
                    'destination_field_name': 'input_pokemon'
                }]
            }]
        },
        # pokemon-sorter -> pokemon-vis
        {
            'destination_name': "pokemon-vis",
            'sources': [{
                'source_name': "pokemon-sorter",
                'mapping': [{
                    "source_field_name": 'output_pokemon',
                    'destination_field_name': 'input_pokemon'
                }]
            }]
        },
        # pokemon-vis -> pipeline end
        {
            'destination_name': "pipeline_end",
            'sources': [{
                'source_name': "pokemon-vis",
                'mapping': [{
                    "source_field_name": 'output_pokemon',
                    'destination_field_name': 'output_pokemon'
                }]
            }]
        }
    ]
)

api.pipeline_versions_create(project_name=PROJECT_NAME, pipeline_name=pipeline_name, data=pipeline_template)
```

## All done! Let's close the client properly.


```python
client.close()
```

## Pokemon pipeline done!
If you check in your UbiOps account under pipeline you will find a pokemon-pipeline with our components in it and connected. Let's make a request to it. You can also make a request in the UI with the "create direct request button".

This might take a while since the models will need a cold start as they have never been used before.

## Making a request and exploring further
You can go ahead to the Web App and take a look in the user interface at what you have just built. If you want you can create a request to the pipeline. Just input a few Pokemon names. For example:

```
[
'Pikachu',
'Charmander'
]
```

For any questions, feel free to reach out to us via the customer service portal: https://ubiops.atlassian.net/servicedesk/customer/portals
