# GPU image recognition deployment

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/gpu-tutorial/deployment_package){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/blob/master/ready-deployments/gpu-tutorial/deployment_package){ .md-button .md-button--secondary }

Image recognition is widely used nowadays. And while CPUs are a viable option for this, GPUs are much more efficient in 
certain situations. This tutorial will demonstrate how to deploy the image recognition model in this 
[deployment package](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/gpu-tutorial/deployment_package)
on UbiOps and let it use a GPU for inference. We have put the `deployment.py` here as well for your reference. 

```
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from imageio import imread
import numpy as np


class Deployment:

    def __init__(self, base_directory, context):

        print("Initialising deployment")
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        weights = os.path.join(base_directory, "cnn.h5")
        with tf.device('/gpu:0'):
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

In the `__init__` method of the Deployment class we load the model weights. In the `request` method we call
`model.predict` to actually make the prediction. This structure is similar to the one used in the [prediction model example](../prediction-model/prediction-model.md). Only in this case
the input is an image. In UbiOps images should be passed as files. With `imageio` this image can be loaded
by calling `imread(data['your_input_name'])`.

## Running the example in UbiOps

To deploy this example model to your own UbiOps environment you can log in to the Webapp and create a new deployment 
in the deployment tab. You will be prompted to fill in certain parameters, you can use the following:

| Deployment configuration         |                                                                                                                                                                    |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Name                             | gpu-deployment                                                                                                                                                     |
| Description                      | A GPU image recognition model                                                                                                                                      | 
| Input fields:                    | name = image, datatype = file                                                                                                                                      | 
| Output fields:                   | name = prediction, datatype = integer                                                                                                                              |
|                                  | name = probability, datatype = double precision                                                                                                                    |
| Version name                     | v1                                                                                                                                                                 | 
| Description                      | leave blank                                                                                                                                                        |
| Environment                         | Python 3.11 - Cuda 11                                                                                                                                               | 
| Upload code                      | [deployment zip](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/gpu-tutorial/deployment_package) |
| Deployment machine device        | GPU                                                                                                                                                                |
| Deployment version instance type | 16384 MB + NVIDIA Tesla T4                                                                                                                                         |
| Request retention                | Leave on default settings                                                                                                                                          | 

All other parameters do not need to be changed.

After uploading the code and with that creating the deployment version UbiOps will start deploying. Once your deployment 
version is available you can make requests to it. For this example three handwritten digits are available for testing.

![](./example_image.jpg)
![](./example_image_2.jpg)
![](./example_image_3.jpg)
