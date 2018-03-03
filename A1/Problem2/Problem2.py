# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:53:05 2018

@author: Chenlian Xu & Qianli Ma
"""

import requests
import zipfile
import pandas as pd
import logging
import os
import shutil
import glob
import boto
from boto.s3.connection import Location
from boto.s3.key import Key
import time
import datetime
"""logging file"""

root = logging.getLogger()
root.setLevel(logging.DEBUG)
ch1 = logging.FileHandler('problem2_log.log')
ch1.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch1.setFormatter(formatter)
root.addHandler(ch1)


'''input the parameters'''

print "Please input the S3 Access Key"
accessKey = raw_input()
logging.info("Access Key = %s" % accessKey)

print "Please input the S3 Secret Access Key"
secretAccessKey = raw_input()
logging.info("Secret Access Key = %s" % secretAccessKey)

print "Please input your location"
location = raw_input()
if location not in ['APNortheast', 'APSoutheast', 'APSoutheast2', 'EU', 'EUCentral1', 'SAEast', 'USWest', 'USWest2']:
    location = 'Default'
logging.info("Location = %s" % location)

year_range = range(2003, 2018)

print "Please input the Year"
year = raw_input()
if int(year) not in year_range:
    logging.error("Invalid year. Please enter a valid year between 2003 and 2017.")
    exit()
logging.info("Year = %s", year)

'''validate Amazon account'''

AWS_ACCESS_KEY_ID = accessKey
AWS_SECRET_ACCESS_KEY = secretAccessKey

try:
    s3_connection = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    """The only way to verify AWS credentials is to actually use them to sign a request and see if it works.
    Here we use get_all_regions() to validates the keys which will not return a huge amount of data."""
    region = s3_connection.get_all_regions()
    print ('connected to S3!')

except:
    logging.info("Amazon keys are invalid!")
    exit()

'''cleaned the required directory'''

zip_dir = year + '_zips'
unzipped_dir = year + '_unzipped'
try:
    if not os.path.exists(zip_dir):
        os.makedirs(zip_dir, mode=0o777)
    else:
        shutil.rmtree(os.path.join(os.path.dirname(__file__), zip_dir), ignore_errors=False)
        os.makedirs(zip_dir, mode=0o777)

    if not os.path.exists(unzipped_dir):
        os.makedirs(unzipped_dir, mode=0o777)
    else:
        shutil.rmtree(os.path.join(os.path.dirname(__file__), unzipped_dir), ignore_errors=False)
        os.makedirs(unzipped_dir, mode=0o777)
    logging.info('Directories cleanup completed.')
except Exception as e:
    logging.error(str(e))
    exit()

''' generate the url'''
domain = "http://www.sec.gov/dera/data/Public-EDGAR-log-file-data/"
urls = []
year_range = range(2003, 2017)
month_quarter = {'Qtr1': ['01', '02', '03'], 'Qtr2': ['04', '05', '06'],
                 'Qtr3': ['07', '08', '09'], 'Qtr4': ['10', '11', '12']}

for key, value in month_quarter.items():
    for v in value:
        url = domain + str(year) + '/' + str(key) + '/' + 'log' + str(year) + str(v) + '01.zip'
        logging.info('url for download %s', url)
        urls.append(url)

'''download the urls'''

try:
    for i in range(0, 12):
        month_zip_dir = zip_dir + '/' + str(i) + '.zip'
        month_unzipped_dir = unzipped_dir + '/' + str(i)
        r = requests.get(urls[i],allow_redirects=True)
        open(month_zip_dir, 'wb').write(r.content)
        if os.path.getsize(month_zip_dir) <= 4515: #catching empty file
            os.remove(month_zip_dir)
            logging.warning('Log file %s is empty.', i)
        else:
            logging.info('Log file %s successfully downloaded', i)
            try:
                zip_ref = zipfile.ZipFile(month_zip_dir, 'r')
                for file in zip_ref.namelist():
                    if file.endswith('.csv'):
                        zip_ref.extract(file, unzipped_dir)
                        zip_ref.close()
                        logging.info('Log file %s was successfully unzipped', i)
            except Exception as e:
                logging.error(str(e))
                exit()
except Exception as e:  # Catching file not found
    logging.warning('Log %s not found...Skipping ahead!', i)
    exit()

file_lists = glob.glob(unzipped_dir + "/*.csv")

all_csv_df_dict = {period: pd.read_csv(period) for period in file_lists}
logging.info('All the csv read into individual dataframes')

try:
    for k, v in all_csv_df_dict.items():
        st = all_csv_df_dict[k]
        for key, value in st.items():
            key_drop = {'cik', 'accession', 'ip', 'date', 'time'}
            key_max = {'idx', 'browser', 'code', 'find', 'extention', 'zone'}
            df = pd.DataFrame(st[key])
            null_count = df.isnull().sum()
            logging.info("count of null in  %s is %s" % (key, null_count))
            most_used_value = pd.DataFrame(df.groupby(key).size().rename('cnt')).idxmax()[0]
            if key == "idx":
                incorrect_idx = (~df.isin([0.0, 1.0])).sum()
                logging.info("count of incorrect idx is %s" % incorrect_idx)
                st[key] = st[key].fillna(most_used_value)
                logging.info("fill the null value in column %s with the most used value" % key)
            elif key == "norefer":
                incorrect_norefer = (~df.isin([0.0, 1.0])).sum()
                logging.info("count of incorrect norefer is %s" % incorrect_norefer)
                st[key] = st[key].fillna('1')
                logging.info("fill the null value in column %s with 1" % key)
            elif key == "noagent":
                incorrect_noagent = (~df.isin([0.0, 1.0])).sum()
                logging.info("count of incorrect noagent is %s" % incorrect_noagent)
                st[key] = st[key].fillna('1')
                logging.info( "fill the null value in column %s with 1" % key)
            elif key in key_drop:
                st[key] = st.dropna(subset=[key])
                logging.info("the null in %s is dropped" % key)
            elif key in key_max:
                st[key] = st[key].fillna(most_used_value)
                logging.info("fill the null value in column %s with the most used value" % key)
            elif key == "crawler":
                st[key] = st[key].fillna('0')
                logging.info("fill the null value in column %s with 0" % key)
            elif key == "size":
                st[key] = st[key].fillna(st[key].mean(axis=0))
                logging.info("fill the null value in column %s with the average value" % key)
except Exception as e:
    logging.error(str(e))
    exit()


'''zip the csv and log'''

try:
    dfs = pd.concat(all_csv_df_dict)
    dfs.to_csv('login_data.csv')
    logging.info('All dataframes of csvs are combined and exported as csv: master_csv.csv.')
except Exception as e:
    logging.error(str(e))
    exit()


def zipdir(path, ziph):
    ziph.write(os.path.join('login_data.csv'))
    ziph.write(os.path.join('problem2_log.log'))


zipf = zipfile.ZipFile('Problem2.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('/', zipf)
zipf.close()
logging.info("csv and log files successfully zipped!")

"""Upload the zip file to AWS S3"""

try:
    zipfile = 'Problem2.zip'
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts)
    bucket_name = AWS_ACCESS_KEY_ID.lower() + str(st).replace(" ", "").replace("-", "").replace(":", "").replace(".", "")
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    bucket = conn.create_bucket(bucket_name, location=location)
    print ('bucket created')
    print "Uploading %s to Amazon S3 bucket %s" % (zipfile, bucket_name)

    k = Key(bucket)
    k.key = 'Problem2'
    k.set_contents_from_filename(zipfile)
    print("Zip File successfully uploaded to S3")
except:
    logging.info("Amazon keys are invalid!!")
    exit()













