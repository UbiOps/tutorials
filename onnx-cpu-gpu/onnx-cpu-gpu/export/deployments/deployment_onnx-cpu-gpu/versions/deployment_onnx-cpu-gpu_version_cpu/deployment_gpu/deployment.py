import onnxruntime
from PIL import Image
from resizeimage import resizeimage
import numpy as np
import urllib.request 
import time


def preprocess_image(image):
    orig_img = Image.open(image)
    img = resizeimage.resize_cover(orig_img, [224,224], validate=False)
    img_ycbcr = img.convert('YCbCr')
    img_y_0, img_cb, img_cr = img_ycbcr.split()
    img_ndarray = np.asarray(img_y_0)
    
    img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
    img_5 = img_4.astype(np.float32) / 255.0
    return img_5, img_cb, img_cr
    
class Deployment:

    def __init__(self, base_directory, context):
        # Save some variables in the init. This way we can track inference times during the batch request.
        self.total_time = 0
        self.n_requests = 0
        urllib.request.urlretrieve("https://github.com/onnx/models/raw/main/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx", "super_resolution.onnx")
        print(onnxruntime.get_device())
        self.ort_session = onnxruntime.InferenceSession("super_resolution.onnx",providers = ['CUDAExecutionProvider'])

        print(self.ort_session.get_providers())
        print("Initialising My Deployment")

    def request(self, data):

        start = time.time()

        input_image, img_cb, img_cr = preprocess_image(data['image'])
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_image} 


        ort_outs = self.ort_session.run(None, ort_inputs )

        img_out_y = ort_outs[0]
        img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
        # get the output image follow post-processing step from PyTorch implementation
        final_img = Image.merge(
            "YCbCr", [
                img_out_y,
                img_cb.resize(img_out_y.size, Image.BICUBIC),
                img_cr.resize(img_out_y.size, Image.BICUBIC),
            ]).convert("RGB")

        final_img.save("image_out.jpg")
        
        # Caculate the average processing time
        self.n_requests = self.n_requests +1
        end = time.time()
        self.total_time = self.total_time + (end - start)
        print(self.total_time/self.n_requests)
        return {
            "output": "image_out.jpg"
        }


