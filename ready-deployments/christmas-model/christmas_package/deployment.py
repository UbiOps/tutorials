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