"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""

import pandas as pd
import numpy as np
import os
import datetime
import pytz
from uuid import uuid4
from joblib import load
from arthurai.core.decorators import log_prediction
from arthurai import ArthurAI


class Deployment:

    def __init__(self, base_directory, context):
        """
        Initialisation method for the deployment. It can for example be used for loading modules that have to be kept in
        memory or setting up connections. Load your external model files (such as pickles or .h5 files) here.

        :param str base_directory: absolute path to the directory where the deployment.py file is located
        :param dict context: a dictionary containing details of the deployment that might be useful in your code.
            It contains the following keys:
                - deployment (str): name of the deployment
                - version (str): name of the version
                - input_type (str): deployment input type, either 'structured' or 'plain'
                - output_type (str): deployment output type, either 'structured' or 'plain'
                - environment (str): the environment in which the deployment is running
                - environment_variables (str): the custom environment variables configured for the deployment.
                    You can also access those as normal environment variables via os.environ
        """

        print("Initialising xgboost model")

        XGBOOST_MODEL = os.path.join(base_directory, "xgboost_model.joblib")
        self.model = load(XGBOOST_MODEL)

        with open("arthur-model-id.txt", 'r') as f:
            print("Initializing Arthur connection")
            self.arthur_model = ArthurAI().get_model(f.read())
            print("Successfully retrieved Arthur model")


    def request(self, data):
        """
        Method for deployment requests, called separately for each individual request.

        :param dict/str data: request input data. In case of deployments with structured data, a Python dictionary
            with as keys the input fields as defined upon deployment creation via the platform. In case of a deployment
            with plain input, it is a string.
        :return dict/str: request output. In case of deployments with structured output data, a Python dictionary
            with as keys the output fields as defined upon deployment creation via the platform. In case of a deployment
            with plain output, it is a string. In this example, a dictionary with the key: output.
        """
        print('Loading data')
        batch = pd.read_csv(data['data']).astype({'id': str})
        batch_id = str(uuid4()).split('-')[-1]

        print("Predictions being made")
        batch['price'] = self.model.predict(batch.drop(columns=['id']).to_numpy())

        print("Sending batch to Arthur")
        inference_data = [{'inference_timestamp': datetime.datetime.now(pytz.utc),
                           'partner_inference_id': row['id'],
                           'batch_id': batch_id,
                           'inference_data': {k: row[k] for k in row.keys() if k != 'id'}}
                           for row in batch.to_dict(orient='records')]
        self.arthur_model.send_inferences(inference_data)
        
        # Writing the prediction to a csv for further use
        print('Writing prediction to csv')
        batch['price'].to_csv('prediction.csv', header = ['house_prices'], index_label= 'index')
        
        return {
            "prediction": 'prediction.csv'
        }
