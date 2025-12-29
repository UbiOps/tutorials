import cv2
import onnxruntime as ort
import argparse
import numpy as np
import sys
import os
from box_utils import predict
from PIL import Image
import base64
import io


def preprocess_image(img_base64):
    """
    The image will be resized using OpenCV to a resolution of 224x224 pixels.
    """

    img = Image.open(io.BytesIO(base64.b64decode(str(img_base64))))
    img_arr = np.asarray(img)

    return img_arr


def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes


# crop image
def cropImage(image, box):
    num = image[box[1]:box[3], box[0]:box[2]]
    return num


# face detection method
def faceDetector(orig_image, face_detector, threshold = 0.7):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs


def ageClassifier(orig_image, age_classifier):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = age_classifier.get_inputs()[0].name
    ages = age_classifier.run(None, {input_name: image})
    age = round(sum(ages[0][0] * list(range(0, 101))), 1)
    return age


class Deployment:

    def __init__(self, base_directory):
        """
        Initialisation method for the deployment. This will be called at start-up of the model in UbiOps.

        :param str base_directory: absolute path to the directory where this file is located.
        """

        onnx_model = os.path.join(base_directory, "vgg_ilsvrc_16_age_imdb_wiki.onnx")
        face_detector_onnx = os.path.join(base_directory, "version-RFB-320.onnx")

        self.age_classifier = ort.InferenceSession(onnx_model)
        self.face_detector = ort.InferenceSession(face_detector_onnx)

    def request(self, data):

        original_image = preprocess_image(data['photo'])
        boxes, labels, probs = faceDetector(original_image, self.face_detector)

        ages = []
        for i in range(boxes.shape[0]):
            box = scale(boxes[i, :])
            cropped = cropImage(original_image, box)
            # gender = genderClassifier(cropped)
            ages.append(ageClassifier(cropped, self.age_classifier))

        # Here we return an integer with the estimated age
        return {'age': int(ages[0])}
