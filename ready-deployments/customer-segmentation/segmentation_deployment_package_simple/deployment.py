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