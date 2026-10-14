"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""


class Deployment:

    def __init__(self, base_directory, context):

        print("Initialising deployment")

    def request(self, data):

        print("Processing request")
        # Retrieve the user input from the data dictionary
        user_input = data['number']

        # Process the input
        number_multiplied = user_input * 2

        # Return the output
        return {'number_multiplied': number_multiplied}
