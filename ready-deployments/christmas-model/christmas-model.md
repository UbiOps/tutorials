# Christmas card maker with OpenCV

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/christmas-model/christmas_package){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/tree/master/ready-deployments/christmas-model/christmas_package){ .md-button .md-button--secondary }

For Christmas we made an example deployment that takes an image as input and returns a Christmas card version of the
image. The deployment uses OpenCV to detect faces in the image and uses that to give everyone a Santa hat. It also adds
some general Christmas decorations to the image.

## The Christmas deployment

This deployment takes an image `input_image` as input, and returns a Christmas card image `festive_image`. We have put
the `deployment.py` here for your reference:

```python
import cv2
import numpy as np
import cvzone

class Deployment:

    def __init__(self, base_directory, context):
        print("Initialising My Deployment")
        # Loading overlay images and classifier
        self.christmas_hat_image = cv2.imread("christmas_hat2.png", cv2.IMREAD_UNCHANGED)
        self.christmas_border = cv2.imread("christmas_border.png", cv2.IMREAD_UNCHANGED)
        self.christmas_decoration = cv2.imread("christmas_decoration.png", cv2.IMREAD_UNCHANGED)
        self.cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def request(self, data):
        print("Processing request for My Deployment")
        print("Reading input image.")
        frame = cv2.imread(data["input_image"])

        # Getting dimensions of the base image
        try:
            height, width, channels = frame.shape
        except:
            height, width = frame.shape

        print("Adding Christmas frame and decoration to image.")
        border_resize = cv2.resize(self.christmas_border, (width, height))
        decoration_resize = cv2.resize(self.christmas_decoration, (width, int(height/4)))
        frame = cvzone.overlayPNG(frame, border_resize, [0,0])
        frame = cvzone.overlayPNG(frame, decoration_resize, [0,0])

        print("Detecting faces.")
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray_scale)

        print("Giving everyone a Christmas hat.")
        for (x, y, w, h) in faces:
            if ((w > 0.045*width) and (h > 0.045*height)):

                overlay_resize = cv2.resize(self.christmas_hat_image, (int(w*1.6), int(h*1.6)))

                try:
                    # This is a better offset for the Christmas hat, but if a face is near the top of the screen it
                    # will give an error. So then we use a smaller offset.
                    frame = cvzone.overlayPNG(frame, overlay_resize, [int(x-w/3), int(y-(0.7)*h)])
                except:
                    frame = cvzone.overlayPNG(frame, overlay_resize, [int(x-w/3), int(y-h/2)])
            else:
                continue
        cv2.imwrite("christmas_image.png", frame)

        return {
            "festive_image": "christmas_image.png"
        }

```

The deployment works in 4 steps:

- Resize & overlay Christmas decoration images on the input image
- Detect faces in input image (using Haar cascades)
- Put Christmas hats above the faces
- Save the resulting image and return it as output

## Running the example in UbiOps

To deploy this example model to your own UbiOps environment you can log in to the WebApp and create a new
deployment in the deployment tab. You will be prompted to fill in certain parameters, you can use the
following:

| Deployment configuration | |
|--------------------|--------------|
| Name | christmas-model|
| Description | A Christmas card maker. Accepts PNG or JPEG images.|
| Input fields: | name = input_image, datatype = file |
| Output fields: | name = festive_image, datatype = file |
| Version name | v1 |
| Description | leave blank |
| Environment | Python 3.8 |
| Upload code | [deployment zip](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/christmas-model/christmas_package) _do not unzip!_|
| Advances parameters | Leave on default settings |

After uploading the code, and with that creating the deployment version, UbiOps will start deploying. Once
you're deployment version is available you can make requests to it. You can use any input image with people on it as
long as it is PNG or JPEG.

!!! warning "Enough space for Christmas hats"
    For the model to work correctly there needs to be enough space above each face in the input image to put the 
    Christmas hat. Otherwise the request might fail.
