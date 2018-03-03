# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:53:05 2018

@author: Chenlian Xu & Qianli Ma
"""
from bs4 import BeautifulSoup
import logging
import requests
import os
import zipfile
import csv
import sys
import boto
from boto.s3.key import Key
import time
from boto.s3.connection import Location
"""logging file"""

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch1 = logging.FileHandler('problem1_log.log')
ch1.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch1.setFormatter(formatter)
root.addHandler(ch1)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

'''input the parameters'''

print "Please input the CIK"
cik = raw_input()
logging.info("CIK = %s" % cik)
if cik != '':
    print "Please input the Accession Number"
    accessionNum = raw_input()
    logging.info("Accession Number = %s" % accessionNum)
    if accessionNum == '':
        cik = '0000051143'
        accessionNum = '0000051143-13-000007'
        logging.info("CIK or Accession Number is not provided by. Take default CIK and Accession Number "
                     "of IBM's quarterly financial filing (form 10-Q) in October 2013 instead.")
else:
    cik = '0000051143'
    accessionNum = '0000051143-13-000007'
    logging.info("CIK or Accession Number is not provided by. Take default CIK and Accession Number "
                 "of IBM's quarterly financial filing (form 10-Q) in October 2013 instead.")

print "Please input the S3 Access Key"
accessKey = raw_input()
logging.info("Access Key = %s" % accessKey)

print "Please input the S3 Secret Access Key"
secretAccessKey = raw_input()
logging.info("Secret Access Key = %s" % secretAccessKey)

print "Please input your location"
location = raw_input()
# if location not in ['APNortheast', 'APSoutheast', 'APSoutheast2', 'EU', 'EUCentral1', 'SAEast', 'USWest', 'USWest2']:
#     location = 'DEFAULT'
"""use boto3 instead of boto"""
logging.info("Location = %s" % location)

"""Validate amazon keys"""

AWS_ACCESS_KEY_ID = accessKey
AWS_SECRET_ACCESS_KEY = secretAccessKey

try:
    s3_connection = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    """The only way to verify AWS credentials is to actually use them to sign a request and see if it works.
    Here we use get_all_regions() to validates the keys which will not return a huge amount of data."""
    print ('connected to S3!')

except:
    logging.info("Amazon keys are invalid!")
    exit()

"""Create the URL by CIK and AccessionNum"""

domain = "http://www.sec.gov/Archives/edgar/data/"
cik_striped = cik.lstrip("0")
accno_striped = accessionNum.replace("-", "")
url1 = domain + cik_striped + "/" + accno_striped + "/" + accessionNum + "-index.html"
logging.info("URL generated is %s " % url1)

"""fetch the form's URL of 10q"""

url_10q = ''
try:
    page = requests.get(url1)
    content = page.text
    soup = BeautifulSoup(content, "lxml")
    links = soup.find_all('a')

    for link in links:
        href = link.get('href')
        if "10q.htm" in href:
            url_10q = "https://www.sec.gov/" + href
            logging.info("10-q file URL is %s " % url_10q)
            break
        else:
            url_10q = ""

    if url_10q == "":
        logging.info("10-q file not found!")
        exit()
except:
    logging.warning("Invalid CIK or Accession Number")
    exit()

"""Fetch the tables and write them into csv files"""

if not os.path.exists('extracted_csvs'):
    os.makedirs('extracted_csvs')

page_10q = requests.get(url_10q)
content_10q = page_10q.text
soup = BeautifulSoup(content_10q, "lxml")
tables = soup.select('div table')

for table in tables:
    records = []
    for tr in table.find_all('tr'):
        rowString = []
        for td in tr.findAll('td'):
            p = td.find_all('p')
            if len(p) > 0:
                for ps in p:
                    ps_text = ps.get_text()
                    ps_text = ps_text.encode("utf-8")
                    ps_text = ps_text.replace(' ', '')
                    ps_text = ps_text.replace('—', "-")
                    rowString.append(ps_text)
            else:
                td_text = td.get_text()
                td_text = td_text.encode("utf-8")
                td_text = td_text.replace(' ', '')
                td_text = td_text.replace('—', "-")
                rowString.append(td_text)
        records.append(rowString)
    with open(os.path.join('extracted_csvs', str(tables.index(table)) + 'tables.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(records)

logging.info("Tables successfully extracted to csv form!")

"""Zip the csvs and logs"""

def zipdir(path, ziph, tables):
    for table in tables:
        ziph.write(os.path.join('extracted_csvs', str(tables.index(table)) + 'tables.csv'))
    ziph.write(os.path.join('problem1_log.log'))

zipf = zipfile.ZipFile('Problem1.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('/', zipf, tables)
zipf.close()
logging.info("csv and log files successfully zipped!")

"""Upload the zip file to AWS S3"""

try:
    zipfile = 'Problem1.zip'
    bucket_name = AWS_ACCESS_KEY_ID.lower() + time.strftime("%y%m%d%H%M%S") + '-dump'
    # bucket_name = AWS_ACCESS_KEY_ID.lower() + '-dump'
    conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    bucket = conn.create_bucket(bucket_name, location=Location.DEFAULT)
    print ('bucket created')
    print "Uploading %s to Amazon S3 bucket %s" % (zipfile, bucket_name)

    k = Key(bucket)
    k.key = 'Problem1'
    k.set_contents_from_filename(zipfile)
    print("Zip File successfully uploaded to S3")
except:
    logging.info("Amazon keys or Bucket names are invalid!")
    exit()