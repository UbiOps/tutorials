# Age estimation with a Mendix front end

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/mendix-age-estimation/deployment_package){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/mendix-age-estimation/deployment_package){ .md-button .md-button--secondary }

This deployment is made for running a neural network model developed with ONNX. Be aware that the model weight file is 
not included and needs to be downloaded separately before uploading this deployment package to UbiOps. See the download link below.

This deployment is part of an article: 
[Building a low code app powered by AI](https://ubiops.com/building-a-low-code-app-powered-by-ai/)

Please read the article for more information on how this model was used in practice in the background of a low-code Mendix app.

_Download link for deployment package_: [onnx-deployment](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/mendix-age-estimation/deployment_package){:target="_blank"}.

_Download link for model files_: 

- [face-detector](https://storage.googleapis.com/ubiops/data/Integration%20with%20other%20tools/mendix-age-estimation/mendix-model-files/version-RFB-320.onnx){:target="_blank"}
- [onnx-model](https://storage.googleapis.com/ubiops/data/Integration%20with%20other%20tools/mendix-age-estimation/mendix-model-files/vgg_ilsvrc_16_age_imdb_wiki.onnx){:target="_blank"}

Add these model files to the deployment package before uploading.

## Running the example in UbiOps

Please take a look at the article for more information:
[Building a low code app powered by AI](https://ubiops.com/building-a-low-code-app-powered-by-ai/)

| Deployment configuration | |
|--------------------|--------------|
| Name | age-estimator|
| Description | A model for estimating age|
| Input field: | name = photo, datatype = string (base64 encoded) |
| Output field: | name = age, datatype = int |
| Version name | v1 |
| Description | leave blank |
| Environment | Python 3.11 |
