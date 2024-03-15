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