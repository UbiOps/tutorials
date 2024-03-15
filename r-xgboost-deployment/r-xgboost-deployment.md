# R-XGboost model template

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/r-xgboost-deployment/r-xgboost-deployment){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/r-xgboost-deployment/r-xgboost-deployment){ .md-button .md-button--secondary }

In this example we will show the following:

How to create a deployment that uses a XGboost model written in R to make predictions on the price of houses, using [data from houses in King County, USA dataset](https://kaggle.com/harlfoxem/housesalesprediction) 

## R-XGboost model
 The deployment is configured as follows:

| Deployment configuration | |
|--------------------|--------------|
| Name | r-xgboost-deployment|
| Function | A deployment that predicts house prices|
| Input field: | name: input_data, data_type: File |
| Output field: | name: prediction, data_type: File|
| Version name | v1 |
| Description | r-xgboost-deployment |
| Environment | R 4.0 |


## How does it work?
**Step 1:** Login to your UbiOps account at https://app.ubiops.com/ and create an API token with project editor
admin rights. To do so, click on **Permissions** in the navigation panel and then click on **API tokens**.
Click on **[+]Add token** to create a new token.

![Creating an API token](../pictures/create-token.gif)

Give your new token a name, save the token in safe place and assign the following roles to the token: project editor.
These roles can be assigned on project level.

**Step 2:** Download the [r-xgboost-tutorial](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/r-xgboost-deployment/r-xgboost-deployment){:target="_blank"}. folder and open r-xgboost-template.R. In the script you will find a space to enter your API token and the name of your project in UbiOps. Paste the saved API token in the notebook in the indicated spot and enter the name of the project in your UbiOps environment. The project name can be found on the top of your screen in the WebApp. In the image in step 1 the project name is example.

**Step 3** Run the R script r-xgboost-template and everything will be automatically deployed to your UbiOps environment! When the deployment has finished building (this takes about 15 minutes) it is possible to make a request with the dummy_data.csv, that is also in the r-xgboost-tutorial folder. Afterwards you can explore the code in the script or explore the application in the WebApp.

_Download links for the necessary files_: *[r-xgboost-tutorial](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/r-xgboost-deployment/r-xgboost-deployment){:target="_blank"}.*