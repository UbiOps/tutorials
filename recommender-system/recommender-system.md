# Deploying a recommender model

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/recommender-system/recommender-system){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/recommender-system/recommender-system){ .md-button .md-button--secondary }

On this page we will show you the following:
- how to train a recommender model on shopping data using [the Apriori algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm)
- How to deploy that model to UbiOps

Recommender models are everywhere nowadays. At every webshop you will receive suggestions based on products you have viewed or added to your shopping cart. In this notebook we will make such a recommender model that can be used in the backend of a webshop. We will use the Apriori algorithm to find rules that describe associations between different products given 7500 transactions over the course of a week at a French retail store. The dataset can be downloaded [here](https://drive.google.com/file/d/1y5DYn0dGoSbC22xowBq2d4po6h1JxcTQ/view).

If you [download](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/recommender-system/recommender-system){:target="_blank"} and run this entire notebook after filling in your access token, the model is trained and deployed to your UbiOps environment. You can thus check your environment after running to explore. You can also check the individual steps in this notebook to see what we did exactly and how you can adapt it to your own use case.

We recommend to run the cells step by step, as some cells can take a few minutes to finish. You can run everything in one go as well and it will work, just allow a few minutes for building the individual deployments.

First let's install and import all the necessary packages.


```python
!pip install apyori 
!pip install matplotlib 
!pip install numpy 
!pip install pandas 
!pip install ubiops
```


```python
# Import all necessary libraries
import shutil
import os
import ubiops
import numpy as np
import pandas as pd
from apyori import apriori
import pickle
```

## Establishing a connection with your UbiOps environment
Add your API token and project name below. Afterwards we initialize the client library. This way we can deploy the model to your environment once we have trained it.


```python
# Set up connection to UbiOps
API_TOKEN = '<INSERT API_TOKEN WITH PROJECT EDITOR RIGHTS>' # Make sure this is in the format "Token token-code"
PROJECT_NAME= '<INSERT PROJECT NAME IN YOUR ACCOUNT>'
DEPLOYMENT_NAME='recommender-model'
DEPLOYMENT_VERSION='v1'

client = ubiops.ApiClient(ubiops.Configuration(api_key={'Authorization': API_TOKEN}, 
                                               host='https://api.ubiops.com/v2.1'))
api = ubiops.CoreApi(client)
```

## Load the data and preprocess

In the next cell we take a look at the data via head(). Don't worry about all the NaN values, this has to do with the ype of data we are dealing with. Our csv file contains orders made by customers. These orders vary a lot, some only buy 3 items, whereas others buy 20. Since a dataframe has a fixed size it takes the size of the longest order and fills up the rest with NaNs. In the preprocessing step we filter out these NaN's when we convert the dataframe to alist of lists, the input format the Apriori algorithm needs.


```python
store_data = pd.read_csv('store_data.csv', header=None)
store_data.head()
```


```python
df_shape = store_data.shape
n_of_transactions = df_shape[0]
n_of_products = df_shape[1]

# Converting our dataframe into a list of lists for Apriori algorithm
records = []
for i in range(0, n_of_transactions):
    records.append([])
    for j in range(0, n_of_products):
        if (str(store_data.values[i,j]) != 'nan'):
            records[i].append(str(store_data.values[i,j]))
        else :
            continue
        
```

## Find association rules with Apriori algorithm

Now that the data is ready we can run the apriori algorithm on our data to find association rules. 


```python
# Run the apriori algorithm
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=2, max_length=5)
association_results = list(association_rules)

# Check how many rules were found
print(len(association_results))
```

Now that we have found our association rules we need to use that to build up a small database that we can query for recommendations. What we want is a database that we can query for a certain product, and that returns three suggestions that a consumer of that product might also be interested in. However, not every association rule gives us three items that are frequently bought with the base item. To make sure that every query will return three recommendations, we will recommend the overall most frequently bought products to fill up the gaps. To do so, we will first have to rank all the products based on how frequently they appear in purchases in our dataset.


```python
# Get all the products listed in dataset
# First merge all the columns of the data frame to a data series object
merged = store_data[0]
for i in range(1,n_of_products):
    merged = merged.append(store_data[i])

# Then rank all the unique products
ranking = merged.value_counts(ascending=False)
# Extract the products in order without their respective count
ranked_products = list(ranking.index)
```

Now that we have a ranking with of the products, and the association rules found by Apriori, we can set up our recommendation rules. In the following cell we also output the support, confidence and lift of the different rules. 

**Support:**
Support refers to the default popularity of an item and can be calculated by finding number of transactions containing a particular item divided by total number of transactions. 

**Confidence:**
Confidence refers to the likelihood that an item B is also bought if item A is bought. It can be calculated by finding the number of transactions where A and B are bought together, divided by total number of transactions where A is bought.

**Lift:**
`Lift(A -> B)` refers to the increase in the ratio of sale of B when A is sold. `Lift(A –> B)` can be calculated by dividing `Confidence(A -> B)` divided by `Support(B)`.


```python
lookup_table = {}
for item in association_results:

    # First index of the inner list contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    to_print = "Rule: "
    arrow = " -> "
    for i in range(len(items)):
        to_print += str(items[i]) + arrow
        
    # If we do not have 3 recommendations for our base product we will
    # suggest top ranked products in addition
    if len(items) < 4:
        items_to_append = items
        i = 0
        while len(items) < 4:
            if ranked_products[i] not in items:
                items_to_append.append(ranked_products[i])
            i += 1
    
    # Add the items to db, with base product separately from the products 
    # that are to be recommended
    lookup_table[items_to_append[0]] =items_to_append[1:]

    print(to_print)

    # Print the support for this association rule
    print("Support: " + str(item[1]))

    # Print the confidence and lift for this association rule

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
```


```python
# The dictionary does not contain recommendations for all products
# In case we don't have a recommendation, the top 3 most frequently bought items 
# need to be suggested. Therefore we need an additional entry in our table
lookup_table['default_recommendation'] = ranked_products[:3]
```


```python
# And now we pickle the dictionary for later use in our deployed model
with open('recommender_deployment_package/lookup_table.pickle', 'wb') as handle:
    pickle.dump(lookup_table, handle)

```

## Deploying recommender model to UbiOps

We have generated our look up table for recommendations, based on the Apriori algorithm. Now we need to deploy a model to UbiOps that outputs recommendations based on this look up table. The deployment we made to do this can be found in the dpeloyment package as deployment.py. It is loaded below so you can take a look at the code.


```python
%%writefile recommender_deployment_package/deployment.py
import os
import pickle


class Deployment:

    def __init__(self, base_directory, context):
        print("Initialising recommender model")

        lookup_table = os.path.join(base_directory, "lookup_table.pickle")
        with open(lookup_table, 'rb') as handle:
            self.lookup_table = pickle.load(handle)

    def request(self, data):
        print('Fetching recommendations')
        input_product = data['clicked_product']
        try:
            recommendation = self.lookup_table[input_product]
        except KeyError:
            recommendation = self.lookup_table['default_recommendation']

        return {
            "recommendation": recommendation
        }


```

## Let's deploy to UbiOps!


```python
# Set up deployment template
deployment_template = ubiops.DeploymentCreate(
    name=DEPLOYMENT_NAME,
    description='Recommmends other products to look at based on clicked product',
    input_type='structured',
    output_type='structured',
    input_fields=[
        {'name':'clicked_product', 'data_type':'string'}
    ],
    output_fields=[
        {'name':'recommendation', 'data_type':'array_string'}
    ],
    labels={'demo': 'recommender-system'}
)

api.deployments_create(
    project_name=PROJECT_NAME,
    data=deployment_template
)

# Create the version
version_template = ubiops.DeploymentVersionCreate(
    version=DEPLOYMENT_VERSION,
    environment='python3-11',
    instance_type_group_name='256 MB + 0.0625 vCPU',
    minimum_instances=0,
    maximum_instances=5,
    maximum_idle_time=1800, # = 30 minutes
    request_retention_mode='none' # We don't need to store the requests in this demo
)

api.deployment_versions_create(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    data=version_template
)

# Zip the deployment package
shutil.make_archive('recommender_deployment_package', 'zip', '.', 'recommender_deployment_package')

# Upload the zipped deployment package
file_upload_result =api.revisions_file_upload(
    project_name=PROJECT_NAME,
    deployment_name=DEPLOYMENT_NAME,
    version=DEPLOYMENT_VERSION,
    file='recommender_deployment_package.zip'
)
```

## All done! Let's close the client properly.


```python
api_client.close()
```
**Note**: This notebook runs on Python 3.11 and uses UbiOps CLient Library 3.15.0. It is used in [this blogpost](https://ubiops.com/how-to-build-and-implement-a-recommendation-system-from-scratch-in-python/).

## Making a request and exploring further
You can go ahead to the Web App and take a look in the user interface at what you have just built. If you want you can create a request to deployment using any product from the original csv file as input, for instance `spaghetti`.  

So there we have it! We have made a recommender model and deployed it to UbiOps. You can use this notebook as inspiration for your own recommender model. 

For any questions, feel free to reach out to us via the customer service portal: https://ubiops.atlassian.net/servicedesk/customer/portals


```python

```
