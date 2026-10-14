"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import os
import pandas as pd
from tensorflow.keras.models import load_model

class Deployment:

    def __init__(self, base_directory, context):
        
        print("Initialising KNN model")
        model_file = os.path.join(base_directory, "tensorflow_model.h5")
        self.model = load_model(model_file)

    def request(self, data):
        # Loading in the data that was sent with the request.
        print('Loading data')
        input_data = pd.read_csv(data['data'])
        
        # Calling the model for a prediction
        print("Prediction being made")
        prediction = self.model.predict(input_data)

        # After the prediction is made you can perform additional processing steps as you please
        # We simply write the prediction to a csv
        print('Writing prediction to csv')
        pd.DataFrame(prediction).to_csv('prediction.csv', header = ['MPG'], index_label= 'index')
        
        return {
            "prediction": 'prediction.csv'
        }
