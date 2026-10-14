# Image recognition deployment

<div class="videoWrapper">

<iframe src="https://youtube.com/embed/uPw05Djo-uU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

</div>

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/image-recognition/mnist_deployment_package){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/ready-deployments/image-recognition/mnist_deployment_package){ .md-button .md-button--secondary }

Image recognition is widely used nowadays. Image recognition apps are fairly straightforward to deploy on UbiOps
and in [this deployment package](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/image-recognition/mnist_deployment_package) you can see an example. It is a model that predicts hand written digits.
It takes a picture of a handwritten digit as input and returns its prediction of what digit it is. We have put the
`deployment.py` here as well for your reference:

```python
import os
from keras.models import load_model
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

In the `__init__` method of the Deployment class we load in the model weights. In the `request` method we call
`model.predict` to actually make the prediction. This structure is similar to the one used in the [prediction model example](../prediction-model/prediction-model.md). Only in this case
the input is an image. With UbiOps images should be passed as files. With `imageio` this image can be loaded
by calling `imread(data['your_input_name'])`.


## Running the example in UbiOps

To deploy this example model to your own UbiOps environment you can log in to the WebApp and create a new
deployment in the deployment tab. You will be prompted to fill in certain parameters, you can use the
following:

| Deployment configuration | |
|--------------------|--------------|
| Name | mnist|
| Description | An image recognition model|
| Input fields: | name = image, datatype = file |
| Output fields: | name = prediction, datatype = integer |
|                | name = probability, datatype = double precision |
| Version name | v1 |
| Description | leave blank |
| Environment | Ubuntu 20.04 + Python 3.9 |
| Upload code | [deployment zip](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/image-recognition/mnist_deployment_package) _do not unzip!_|
| Request retention | Leave on default settings |


The advanced parameters and labels can be left as they are. They are optional.

After uploading the code and with that creating the deployment version UbiOps will start deploying. Once
you're deployment version is available you can make requests to it. For this example, three handwritten digits are available for testing.

![](example_image.jpg)
![](example_image_2.jpg)
![](example_image_3.jpg)
