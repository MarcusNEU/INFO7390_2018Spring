import pandas as pd
import boto3
import pickle
from werkzeug.utils import secure_filename
import time
import os
from flask import flash


ALLOWED_EXTENSIONS = {'csv'}
PICKLED_MODELS = ['svm_model.pkl', 'random_forest_model.pkl']
local_time = time.strftime('%y%m%d-%H%M%S', time.localtime(time.time()))
BUCKET_NAME = 'info7390-2018spring-team2-final-models'
ERROR_METRICS = 'error_metrics.csv'
PICKLED_MODEL_COLUMN_SETS = ['svm_columns.pkl', 'random_forest_columns.pkl']

try:
    S3 = boto3.client('s3', region_name='us-east-1')
except:
    flash("Fail to connect to S3!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def load_column(key):
    try:
        # Load model from S3 bucket
        response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
        # Load pickle model
        column_str = response['Body'].read()
        remained_column = pickle.loads(column_str)
        return remained_column
    except:
        flash("Fail to Load Feature Engineering Results!")


def feature_engineering(data, remained_column):
    data_ = data[remained_column]
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
        flash("Fail to Load Model!")


def data_processing_user(input_file):
    try:
        data = pd.read_csv(input_file)
        total_rows = data.shape[0]
        final_result = []
        amount = data[data.columns[-1]].tolist()
        final_result = amount
        column_name = ['Time', 'Description', 'Type', 'Amount', 'Label']
        for i in range(len(amount)):
            if amount[i] < 5:
                final_result[i] = 'Fraud'
            else:
                final_result[i] = 'Normal'
        data['label'] = amount
        output_rows = []
        for i in range(0,total_rows):
            output_rows.append([data.iat[i, 0], data.iat[i, 1], data.iat[i, 2], data.iat[i, 3], data.iat[i, 4]])
        return column_name, output_rows, total_rows
    except:
        print "The uploaded data cannot be predicted!"




def data_processing(input_file):
    try:
        data = pd.read_csv(input_file)
        total_rows = data.shape[0]
        data_re = data.loc[:, ['Time', 'Amount']]
        column_name = ['Time', 'Amount', 'Label']
        final_result = []
        #column_name = ['support vector machine', 'random forest']
        svm_remained_column = load_column(PICKLED_MODEL_COLUMN_SETS[0])
        svm_data_ = feature_engineering(data, svm_remained_column)
        # Load Model
        svm_model = load_model(PICKLED_MODELS[0])
        # Make prediction
        svm_prediction = svm_model.predict(svm_data_).tolist()
        final_result = svm_prediction
        for i in range(0, total_rows):
            if svm_prediction[i] == 0:
                rf_remained_column = load_column(PICKLED_MODEL_COLUMN_SETS[1])
                rf_data_ = feature_engineering(data, rf_remained_column)
                #Load Model
                rf_model = load_model(PICKLED_MODELS[1])
                # Make prediction
                rf_prediction = rf_model.predict(rf_data_.loc[[i]])
                final_result[i] = rf_prediction.tolist()[0]
        for i in range(len(final_result)):
            if final_result[i] == 0:
                final_result[i] = 'Normal'
            else:
                final_result[i] = 'Fraud'
        data_re['Label'] = final_result
        output_row = []
        for i in range(0, total_rows):
            output_row.append([data_re.iat[i, 0], data_re.iat[i, 1], data_re.iat[i, 2]])
        return column_name, output_row, total_rows
    except:
        flash("The uploaded data cannot be predicted!")


def form_download_file(output_folder, output_row):
    try:
        output_filename = str(local_time) + '_result.' + 'csv'
        output_path = os.path.join(output_folder, output_filename)
        download_file = pd.DataFrame(data=output_row)
        download_file.to_csv(output_path, mode='a',header=None)
        return output_path
    except:
        flash( "Fail to Form Download File!")


