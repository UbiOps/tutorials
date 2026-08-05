import os
import pickle


class Deployment:

    def __init__(self, base_directory, context):
        print("Initialising recommender model")

        lookup_table = os.path.join(base_directory, "lookup_table.pickle")
        with open(lookup_table, 'rb') as handle:
            self.lookup_table = pickle.load(handle)

    def request(self, data):
        print('Fetching recommendations')
        input_product = data['clicked_product']
        try:
            recommendation = self.lookup_table[input_product]
        except KeyError:
            recommendation = self.lookup_table['default_recommendation']

        return {
            "recommendation": recommendation
        }
