"""
The file containing the deployment code is required to be called 'deployment.py' and should contain the 'Deployment'
class and 'request' method.
"""
import snowflake.connector as sf
import os


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

        print("Initialising My Deployment")
        print('Connecting to snowflake database')

        SNOWFLAKE_ACCOUNT = os.environ.get('SNOWFLAKE_ACCOUNT')
        SNOWFLAKE_USERNAME= os.environ.get('SNOWFLAKE_USERNAME')
        SNOWFLAKE_PASSWORD = os.environ.get('SNOWFLAKE_PASSWORD')
        SNOWFLAKE_DATABASE = os.environ.get('SNOWFLAKE_DATABASE')

        try:        
            con = sf.connect(
                user=SNOWFLAKE_USERNAME,
                password=SNOWFLAKE_PASSWORD,
                account=SNOWFLAKE_ACCOUNT,
                database=SNOWFLAKE_DATABASE
            )
            self.cur = con.cursor()

        except Exception as e:
            print('There was a problem connecting to the database!')
            print(e)


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

        # Price limit
        max_price = data['max_price']
        self.cur.execute(f'SELECT * from product where price < {max_price};')
        items = self.cur.fetchall()

        # No affordable items
        if len(items) == 0:
            return {
                'output': "unfortunately you cannot afford any of the items in our shop!"
            }

        # Format items that the user can purchase
        affordable_items = ', '.join(list(map(lambda x: x[0], items)))
        return {
            "output": f"You can afford to buy the following ({affordable_items})"
        }
