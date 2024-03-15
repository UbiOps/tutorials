"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import pandas as pd
import numpy as np
import os
from joblib import load

class Deployment:

    def __init__(self, base_directory, context):
        print("Initialising xgboost model")

        XGBOOST_MODEL = os.path.join(base_directory, "fraud_model.joblib")
        self.model = load(XGBOOST_MODEL)

    def request(self, data):
        print('Loading data')
        input_data = pd.read_csv(data['input'])
        
        print("Prediction being made")
        prediction = self.model.predict(input_data)
        
        # Writing the prediction to a csv for further use
        print('Writing prediction to csv')
        pd.DataFrame(prediction).to_csv('prediction.csv', header = ['Class prediction'], index_label= 'index')
        
        return {
            "output": 'prediction.csv'
        }
