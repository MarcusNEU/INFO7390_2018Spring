import pandas as pd
import boto3
import pickle
from dask.distributed import Client
from werkzeug.utils import secure_filename
import time
import os
from common.custom_expections import BaseError


ALLOWED_EXTENSIONS = {'json'}
BUCKET_NAME = 'akiaj53cehklbfj6cf4q180411152356-dump'
PICKLED_MODELS = ['finalized_linear_model.pkl', 'finalized_random_forest_model.pkl', 'finalized_neural_network_model.pkl']
local_time = time.strftime('%y%m%d-%H%M%S', time.localtime(time.time()))
# list1 = [1,2,3,4,8,21,25,26,27,28,29,30,35,40,45,46,50,51,52,53,55,56,57,58,60,61,62,63,64,65,66,67,68,69,70,71,72,74,75,77,78,79,82,83,84,85,86,87,88,89,90,91,92,93,95,96,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,131,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,150,151,152,153,154,155,157,158,159,160,162,163,164,165,167,168,170,171,173,174,175,176,177,178,180,181,182,183,184,185,186,187,188,189,191,192,193,194,195,196,197,199,201,202,203,204,205,206,207,208,209,210,211,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,237,238,239,240,241,242,244,245,246,247,248,249,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,279,280]
PICKLED_MODEL_COLUMN_SETS = ['lm_index.pkl', 'tree_index.pkl', 'nn_index.pkl']

try:
    S3 = boto3.client('s3', region_name='us-east-1')
except:
    raise BaseError(code=500, message="Fail to connect to S3!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def feature_engineering(data, remained_column):
    data_ = data.iloc[:, [i-1 for i in remained_column]]
    return data_


def load_model(key):
    try:
        # Load model from S3 bucket
        response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
        # Load pickle model
        model_str = response['Body'].read()
        model = pickle.loads(model_str)
        return model
    except:
        raise BaseError(code=500, message="Fail to Load Model!")


def load_column(key):
    try:
        # Load model from S3 bucket
        response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
        # Load pickle model
        column_str = response['Body'].read()
        model = pickle.loads(column_str)
        return model
    except:
        raise BaseError(code=500, message="Fail to Load Model!")


def data_processing(input_file, upload_folder):
    if not input_file:
        raise BaseError(code=500, message="No File Uploaded!")
    elif not allowed_file(input_file.filename):
        raise BaseError(code=500, message="File Form Error! Only json File is Accepted!")
    else:
        try:
            file_uploaded_name = secure_filename(input_file.filename)
            suffix = file_uploaded_name.rsplit('.', 1)[1]
            new_filename = str(local_time) + '.' + suffix
            data = pd.read_json(input_file, typ='frame', numpy=True, orient='records')
            total_rows = data.shape[0]
            upload_save_path = os.path.join(upload_folder, new_filename)
            input_file.save(upload_save_path)
            output_column = ['linear model', 'random forest', 'neural network']
            targets = []
            for i in range(0, len(PICKLED_MODELS)):
                remained_column = [int(c) for c in load_column(PICKLED_MODEL_COLUMN_SETS[i])]
                data_ = feature_engineering(data, remained_column)
                # Load Model
                model = load_model(PICKLED_MODELS[i])
                # Make prediction
                if total_rows > 10:
                    client = Client(processes=False)
                    prediction = client.submit(model.predict, data_).result().tolist()
                else:
                    prediction = model.predict(data_).tolist()
                targets.append(prediction)
            output_row = []
            for i in range(0, total_rows):
                output_row.append([targets[0][i], targets[1][i], targets[2][i]])
            return output_column, output_row, total_rows
        except:
            raise BaseError(code=500, message="The uploaded data cannot be predicted!")


def form_download_file(output_folder, output_column, output_row):
    try:
        output_filename = str(local_time) + '_result.' + 'csv'
        output_path = os.path.join(output_folder, output_filename)
        download_file = pd.DataFrame(columns=output_column, data=output_row)
        download_file.to_csv(output_path)
        return output_path
    except:
        raise BaseError(code=500, message="Fail to Form Download File!")

