# Customer segmentation

[Download notebook :fontawesome-solid-download:](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/customer-segmentation){ .md-button .md-button--primary } [View source code :fontawesome-brands-github: ](https://github.com/UbiOps/tutorials/tree/master/ready-deployments/customer-segmentation){ .md-button .md-button--secondary }

Customer segmentation is a common application of Data Science. We have an example deployment that segments customers
based on an RFM analysis. We have two versions:

- [A simple version](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/customer-segmentation/segmentation_deployment_package_simple) that takes an Excel file as input, and that returns a CSV file.
- [A more advanced version](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/customer-segmentation/segmentation_deployment_package_advanced) that reads the data from a Google Sheet, and writes the outcomes to the same Google sheet.

## Simple RFM deployment

This deployment takes an excel file `file` as input, and returns an excel file `segmentation`. We have put the
`deployment.py` here for your reference:

```python
import os
import pandas as pd


class Deployment:

    def __init__(self, base_directory, context):

        print('Initialising the connection to the google drive')
                 

    def request(self, data):

        print('Getting the requested file')
        # Transforming data into a Pandas DataFrame
        data_df = pd.read_excel(data['file'])

        # RFM analyis
        print('Performing RFM analysis')
        data_df['TotalPrice'] = data_df['Quantity'].astype(int) * data_df['UnitPrice'].astype(float)
        data_df['InvoiceDate'] = pd.to_datetime(data_df['InvoiceDate'])

        rfm= data_df.groupby('CustomerID').agg({'InvoiceDate': lambda date: (date.max() - date.min()).days,
                                                'InvoiceNo': lambda num: len(num),
                                                'TotalPrice': lambda price: price.sum()})

        # Change the name of columns
        rfm.columns=['recency','frequency','monetary']

        # Computing Quantile of RFM values
        rfm['recency'] = rfm['recency'].astype(int)
        rfm['r_quartile'] = pd.qcut(rfm['recency'].rank(method='first'), 4, ['1','2','3','4']).astype(int)
        rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1']).astype(int)
        rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1']).astype(int)

        # Concatenating the quantile numbers to get the RFM score
        rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)

        # Sort the outcomes
        print('Sorting customers')
        sorted_customers = rfm.sort_values('RFM_Score', ascending=True)
        sorted_customers.to_csv('segmentation_results.csv', index=True)
        
        return {
            'segmentation': 'segmentation_results.csv'
        }

```

In the `request` method we read in the Excel file as a DataFrame and we assign an RFM score to each customer based on
how frequently, how recently and for how much money they purchased. The outcome is sorted and written to a csv, which
we return as output.


### Running this example in UbiOps

To deploy this example model to your own UbiOps environment you can log in to the WebApp and create a new
deployment in the deployment tab. You will be prompted to fill in certain parameters, you can use the
following:

| Deployment configuration | |
|--------------------|--------------|
| Name | rfm-analysis|
| Description | A customer segmentation model based on RFM.|
| Input fields: | name = file, datatype = file |
| Output fields: | name = segmentation, datatype = file |
| Version name | v1 |
| Description | leave blank |
| Environment | Python 3.8 |
| Upload code | [deployment zip](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/customer-segmentation/segmentation_deployment_package_simple) _do not unzip!_|
| Advanced settings | Leave on default settings |

After uploading the code and with that creating the deployment version UbiOps will start deploying. Once
you're deployment version is available you can make requests to it. For this example, you can use [this example
dataset from Kaggle](https://kaggle.com/mrmining/online-retail).

## Advanced RFM deployment

This deployment takes a filename as input, and returns nothing. Instead, the deployment retrieves data from a Google
Sheet and writes the results of the analysis to that same sheet.

!!! info "Google Sheets connection"
    For this deployment to work, you will need to have [this dataset](https://kaggle.com/mrmining/online-retail)
    saved in your own Google Drive, and you will need a service account that has access to that sheet. For information
    on this see [Pygsheet documentation](https://pygsheets.readthedocs.io/en/stable/authorization.html#service-account).

The `deployment.py` for this more advanced version of the RFM analysis can be seen below.

```python
import os
import json
from google.oauth2 import service_account
import pygsheets
import pandas as pd


class Deployment:

    def __init__(self, base_directory, context):

        print('Initialising the connection to the google drive')
        self.gc = None

        SCOPES = ('https://googleapis.com/auth/spreadsheets', 'https://googleapis.com/auth/drive')
        service_account_info = json.loads(os.environ['credentials'])
        my_credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)

        try:
            self.gc = pygsheets.authorize(custom_credentials=my_credentials)
            print('Established succesfull connection')
        except Exception as e:
            print('Connection failed, ', e.__class__, 'occurred.')
            

    def request(self, data):

        print('Getting the requested file')
        spreadsheet = self.gc.open(data['filename'])
        sheet_data = spreadsheet[0]

        # Transforming data into a Pandas DataFrame
        data_df = sheet_data.get_as_df()

        # RFM analyis
        print('Performing RFM analysis')
        data_df['TotalPrice'] = data_df['Quantity'].astype(int) * data_df['UnitPrice'].astype(float)
        data_df['InvoiceDate'] = pd.to_datetime(data_df['InvoiceDate'])

        rfm= data_df.groupby('CustomerID').agg({'InvoiceDate': lambda date: (date.max() - date.min()).days,
                                                'InvoiceNo': lambda num: len(num),
                                                'TotalPrice': lambda price: price.sum()})

        # Change the name of columns
        rfm.columns=['recency','frequency','monetary']

        # Computing Quantile of RFM values
        rfm['recency'] = rfm['recency'].astype(int)
        rfm['r_quartile'] = pd.qcut(rfm['recency'].rank(method='first'), 4, ['1','2','3','4']).astype(int)
        rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1']).astype(int)
        rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1']).astype(int)

        # Concatenating the quantile numbers to get the RFM score
        rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)

        # Sort the outcomes
        print('Sorting customers')
        sorted_customers = rfm.sort_values('RFM_Score', ascending=True)
        
        # Insert data into the google sheet in a separate tab
        print('Inserting data into the google sheet')
        sheet_title = "RFM_results"

        try:
            sh = spreadsheet.worksheet_by_title(sheet_title)
        except:
            print('Worksheet does not exist, adding new sheet')
            spreadsheet.add_worksheet(sheet_title)
            sh = spreadsheet.worksheet_by_title(sheet_title)
        finally:
            sh.set_dataframe(sorted_customers, 'A1', copy_index = True)
            sh.update_value('A1', 'CustomerID')
            print('Data inserted successfully')
        
        return None

```

In the `__init__` method we set up a connection to the google drive where the sheet resides. In the `request` method we
retrieve the Google Sheet as a DataFrame and we assign an RFM score to each customer based on how frequently, how
recently and for how much money they purchased. The outcome is sorted and written to the same Google Sheet, in a
separate tab called `RFM_results`.

### Running this example in UbiOps

To deploy this example model to your own UbiOps environment you can log in to the WebApp and create a new
deployment in the deployment tab. You will be prompted to fill in certain parameters, you can use the
following:

| Deployment configuration | |
|--------------------|--------------|
| Name | rfm-analysis|
| Description | A customer segmentation model based on RFM.|
| Input fields: | name = filename, datatype = string |
| Output fields: | leave blank |
| Version name | v1 |
| Description | leave blank |
| Environment | Python 3.8 |
| Upload code | [deployment zip](https://download-github.ubiops.com/#!/home?url=https://github.com/UbiOps/tutorials/tree/master/ready-deployments/customer-segmentation/segmentation_deployment_package_advanced) _do not unzip!_|
| Advanced settings | Expand |
| Environment variables| name = credentials, value = your [JSON credential string](https://pygsheets.readthedocs.io/en/stable/authorization.html#service-account)|

After uploading the code and with that creating the deployment version UbiOps will start deploying. Once
you're deployment version is available you can make requests to it. For this example, you will need to use the name of
your Google Sheet as input (e.g. `OnlineRetail`). Afterwards you should see a new tab in your Google Sheet with the
results.
