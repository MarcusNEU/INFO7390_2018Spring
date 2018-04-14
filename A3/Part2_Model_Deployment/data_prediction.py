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
PICKLED_MODEL_COLUMN_SETS = ['lm_index.pkl', 'tree_index.pkl', 'nn_index.pkl']
ERROR_METRICS = 'error_metrics.csv'

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
        remained_column = pickle.loads(column_str)
        return remained_column
    except:
        raise BaseError(code=500, message="Fail to Load Feature Engineering Results!")


def get_metrics(key):
    try:
        response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
        metrics = response['Body'].read()
        return metrics
    except:
        raise BaseError(code=500, message="Fail to Load Error Metrics!")


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
            column_name = ['linear model', 'random forest', 'neural network']
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
            return column_name, output_row, total_rows
        except:
            raise BaseError(code=500, message="The uploaded data cannot be predicted!")


def unpickle_error_metrics():
    error_metrics = get_metrics(ERROR_METRICS)
    metrics_list = error_metrics.split('\n')
    column_name = ['Model', 'r^2_test', 'rmse_test']
    output_row = []
    for i in range(1,4):
        output_row.append(
            [metrics_list[i].split(',')[0], metrics_list[i].split(',')[5], metrics_list[i].split(',')[7]])
    return column_name, output_row


def form_download_file(output_folder, output_column, output_row, metrics_column, metrics_row):
    try:
        output_filename = str(local_time) + '_result.' + 'csv'
        output_path = os.path.join(output_folder, output_filename)
        error_metrics = pd.DataFrame(columns=metrics_column, data=metrics_row)
        download_file = pd.DataFrame(columns=output_column, data=output_row)
        print error_metrics
        print download_file
        error_metrics.to_csv(output_path)
        download_file.to_csv(output_path, mode='a', header=True)
        return output_path
    except:
        raise BaseError(code=500, message="Fail to Form Download File!")

