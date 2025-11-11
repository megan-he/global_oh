import os
import datetime
import time
import pandas as pd
import requests
import subprocess
import json
import boto3
import multiprocessing

# Import credentials
import creds
from botocore.exceptions import ClientError

'''
Download TROPOMI data from Copernicus S5P data hub using s3. Modified from Ruijun Dang.
'''

s3 = None
def initialize():
    global s3
    endpoint_url = "https://eodata.dataspace.copernicus.eu"
    s3 = boto3.client(
        's3',
        aws_access_key_id=creds.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=creds.AWS_SECRET_ACCESS_KEY,
        endpoint_url=endpoint_url
    )

# Follow stack overflow directions to download over multiple cores.
# Copernicus specifies that the maxnumber of concurrent connections is 4.
def download_from_s3(args):
    s3_path, final_path = args
    s3_path = s3_path[8:]
    bucket_name = "eodata"
    file =  s3_path.split("/")[-1]
    object_key = s3_path
    local_file_path = final_path + "/" + file
    
    retries = 5
    delay = 1
    for attempt in range(retries):
        try:
            s3.download_file(bucket_name, object_key, local_file_path)
            print(f"Downloaded {file} successfully.")
            break
        except ClientError as e:
            if e.response['Error']['Code'] == '429':
                print(f"Rate limit exceeded, retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Failed to download {file}: {e}")
                break

def download_tropomi_data(species):

    saving_path = f'/n/holylfs05/LABS/jacob_lab/Users/mhe/Obs_data/TROPOMI/{species}/'

    # settings for the data filter
    data_collection = "SENTINEL-5P"
    data_filter = f'OFFL_L2__{species}'
    months = pd.date_range(start='2024-01-01', end='2025-01-01', freq='MS') # end date should be last month you want + 1 month
    # aoi = "POLYGON((-180 -90, -180 90, 180 90, 180 -90,-180 -90))'"

    for start, end in zip(months[:-1], months[1:]):
        start_date = start.strftime('%Y-%m-%d')
        end_date = end.strftime('%Y-%m-%d')
        yyyymm = start.strftime('%Y%m')

        print(f"Downloading {species} for {yyyymm}")
        print("Start:", start_date, "| End:", end_date)
        
        # searching the info of the data we want
        json_list = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' \
        and contains(Name,'{data_filter}') \
        and ContentDate/Start gt {start_date}T00:00:00.000Z \
        and ContentDate/Start lt {end_date}T00:00:00.000Z&$count=True&$top=500").json()
        print('total number of observations that match the filters: ',json_list['@odata.count'])
        
        # save the info in the lists
        data_info = pd.DataFrame.from_dict(json_list['value']).sort_values(by='Name')
        s3_list = data_info['S3Path']
        
        # before downloading, create the final saving path
        final_path = os.path.join(saving_path, yyyymm)
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        
        # downloading the data using above lists
        args = [(s3_path, final_path) for s3_path in s3_list]
        with multiprocessing.Pool(4, initialize) as pool:
            pool.map(download_from_s3, args)
            pool.close()
            pool.join()
            
        # for next circle
        # start_date = end_date
        # break


# Download
species = 'HCHO'
download_tropomi_data(species)