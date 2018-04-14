#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:21:16 2018

@author: chenlianxu & qianli ma
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 17:46:12 2018

@author: chenlianxu & qianli ma
"""

import boto
from boto.s3.key import Key
import time
from boto.s3.connection import Location
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import zipfile
import glob
import shutil
import pandas as pd
import requests
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import airflow
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


blog_dir = '/Users/chenlianxu/Desktop/blog_data'
url_ = 'https://s3.amazonaws.com/assignment3datasets/BlogFeedback.zip'
zip_dir_ = blog_dir + '/blog.zip'
unzipped_dir_ = blog_dir+'/blog'
error_metric = pd.DataFrame({'rmse_train':[], 
                             'rmse_test': [],
                             'mae_train': [],
                             'mae_test':[],
                             'mape_train':[],
                             'mape_test':[],
                                'r_train':[],
                                'r_test':[]})


def cal_metric(modelname,model,X_train, y_train, X_test, y_test):
    global error_metric 
    y_train_pre = model.predict(X_train)
    y_test_pre = model.predict(X_test)
    
    rmse_train = math.sqrt(mean_squared_error(y_train, y_train_pre))
    rmse_test = math.sqrt(mean_squared_error(y_test, y_test_pre))
    
    mae_train = mean_absolute_error(y_train,  y_train_pre)
    mae_test = mean_absolute_error(y_test,  y_test_pre)
     
    mape_train = np.mean(np.abs((y_train - y_train_pre) / y_train)) * 100
    mape_test = np.mean(np.abs((y_test - y_test_pre) / y_test)) * 100
    
    r_train = r2_score(y_train, y_train_pre)
    r_test = r2_score(y_test, y_test_pre)
    
    error_metric_local = pd.DataFrame({'Model':[modelname],
                                      'rmse_train':[rmse_train], 
                                        'rmse_test': [rmse_test],
                                        'mae_train': [mae_train],
                                       'mae_test':[mae_test],
                                        'mape_train':[mape_train],
                                       'mape_test':[mape_test],
                                       'r_train':[r_train],
                                       'r_test':[r_test]})
    
    
    error_metric = pd.concat([error_metric,error_metric_local])
    return error_metric


def clean_dir(download_dir,**kwargs):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir, mode=0o777)
    else:
        shutil.rmtree(os.path.join(os.path.dirname(__file__), download_dir), ignore_errors=False)
        os.makedirs(download_dir, mode=0o777)

def download_zip(url, zip_dir, unzipped_dir, **kwargs):
    resp = requests.get(url)
    open(zip_dir, 'wb').write(resp.content)
    zip_ref = zipfile.ZipFile(zip_dir, 'r')
    for file in zip_ref.namelist():
        zip_ref.extract(file, unzipped_dir)
    zip_ref.close()


def read_df(unzipped_dir, **kwargs):
    filename_test = glob.glob(unzipped_dir + '/*test*.csv')
    list_ = []
    for file in filename_test:
        df = pd.read_csv(file, header=None)
        list_.append(df)
    df_test = pd.concat(list_)
    df_train = pd.read_csv(unzipped_dir + '/blogData_train.csv',header=None)
    df_train.columns = [str(i) for i in range(1,282)]
    df_test.columns = [str(i) for i in range(1,282)]
    return df_train, df_test
    
#def eda()
    
def feture_engineer(**kwargs):
    
    #drop variable with low variance
    df_train, df_test = kwargs['ti'].xcom_pull(task_ids='read')
    selector = VarianceThreshold()
    selector.fit_transform(df_train)
    #create mask for var
    mask_var = selector.get_support()
    
    #drop variable that are highly correlated
    corr_matrix = df_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    #create mask for cor
    mask_corr = ~np.in1d(df_train.columns, to_drop)
    not_drop = df_train.columns[np.logical_and(mask_corr,mask_var)]
    drop = df_train.columns[~np.logical_and(mask_corr,mask_var)]
 
    df_train = df_train.drop(drop, axis=1)
    df_test = df_test.drop(drop, axis=1)
    with open(blog_dir + "/not_drop.pkl", "wb") as fp:  
        pickle.dump(not_drop, fp, protocol = 2)
    return df_train, df_test


#test
def train_model(**kwargs):
    #train linear model and pickle
    df_train, df_test = kwargs['ti'].xcom_pull(task_ids='feature_engineer')
    X_train = df_train.loc[:, df_train.columns != '281']
    y_train = df_train['281']
    X_test = df_test.loc[:,df_test.columns != '281']
    y_test = df_test['281']
    
    
    #train linear model and pickle
    lm_index = X_train.columns.tolist()
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    cal_metric('linear', lm ,X_train, y_train, X_test, y_test)
    filename_l = blog_dir + '/finalized_linear_model.pkl'
    pickle.dump(lm,open(filename_l,'wb'), protocol = 2)
    
    lm_index_file = blog_dir + '/lm_index.pkl'
    with open(lm_index_file, "wb") as fp:  
        pickle.dump(lm_index, fp, protocol = 2)
    
    
    #train random forest and pickle
    tree_index = ['1','2','52','55','61','62']
    rf = RandomForestRegressor()
    X_train_rf = X_train[tree_index]
    X_test_rf = X_test[tree_index]
    rf.fit(X_train_rf, y_train)
    cal_metric('random forest', rf ,X_train_rf, y_train, X_test_rf, y_test)
    filename_rf = blog_dir + '/finalized_random_forest_model.pkl'
    with open(filename_rf, 'wb') as fp:
        pickle.dump(rf,fp, protocol = 2)
    
    tree_index_file = blog_dir + '/tree_index.pkl'
    with open(tree_index_file, "wb") as fp:  
        pickle.dump(tree_index, fp, protocol = 2)
    
    
    #train nn
    nn_index = X_train.columns.tolist()
    nn = MLPRegressor(activation='logistic', alpha=0.1, batch_size=600, beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(50,), learning_rate='constant',
       learning_rate_init=0.1, max_iter=10000, momentum=0.01,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
    
    nn.fit(X_train, y_train)
    cal_metric('network', nn ,X_train, y_train, X_test, y_test)
    filename_nn = blog_dir + '/finalized_neural_network_model.pkl'
    pickle.dump(nn,open(filename_nn,'wb'), protocol = 2) 
    
    nn_index_file = blog_dir + '/nn_index.pkl'
    with open(nn_index_file, "wb") as fp:  
        pickle.dump(nn_index, fp, protocol = 2)
    
    
    ##save error_metrics
    
    error_metric.to_csv(blog_dir + '/error_metrics.csv', index = False)
    
    
    
    
def connect():
    accessKey = 'your amazon accessKey'
    secretAccessKey='your amazon secretAccessKey'
    AWS_ACCESS_KEY_ID = accessKey
    AWS_SECRET_ACCESS_KEY = secretAccessKey
        
    try:
        error_file = blog_dir + '/error_metrics.csv'
        linear_file = blog_dir + '/finalized_linear_model.pkl'
        random_file = blog_dir + '/finalized_random_forest_model.pkl'
        nn_file = blog_dir + '/finalized_neural_network_model.pkl'
        not_drop_file = blog_dir + '/not_drop.pkl'
        tree_index_file = blog_dir + '/tree_index.pkl'
        lm_index_file = blog_dir + '/lm_index.pkl'
        nn_index_file = blog_dir + '/nn_index.pkl'
        
        
        
        bucket_name ='pipelinefiletest7'+ AWS_ACCESS_KEY_ID.lower() +  time.strftime("%y%m%d%H%M%S")+ '-dump'
        conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        bucket = conn.create_bucket(bucket_name, location=Location.DEFAULT)
        print ('bucket created')
    
        k1 = Key(bucket)
        k1.key = 'finalized_linear_model.pkl'
        k1.set_contents_from_filename(linear_file)
        
        k2 = Key(bucket)
        k2.key = 'finalized_random_forest_model.pkl'
        k2.set_contents_from_filename(random_file)
        
        
        k3 = Key(bucket)
        k3.key = 'finalized_neural_network_model.pkl'
        k3.set_contents_from_filename(nn_file)
        
        
        k4 = Key(bucket)
        k4.key = 'index.pkl'
        k4.set_contents_from_filename(not_drop_file)
        
        
        k5 = Key(bucket)
        k5.key = 'error_metric.csv'
        k5.set_contents_from_filename(error_file)
        
        k6 = Key(bucket)
        k6.key = 'tree_index.pkl'
        k6.set_contents_from_filename(tree_index_file)
        
        
        k7 = Key(bucket)
        k7.key = 'lm_index.pkl'
        k7.set_contents_from_filename(lm_index_file)
        
        k8 = Key(bucket)
        k8.key = 'nn_index.pkl'
        k8.set_contents_from_filename(nn_index_file)
        
        
        print(" File successfully uploaded to S3")
    except:
        exit()

args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2)
}
    

dag = DAG('blog_data_pipeline_10', default_args=args, schedule_interval= '@once' )

t0 = PythonOperator(
    task_id='clean',
    python_callable=clean_dir,
    provide_context = True,  
    op_kwargs = {'download_dir':blog_dir},
    dag=dag)

t1 = PythonOperator(
    task_id='download',
    python_callable=download_zip,
    provide_context = True,  
    op_kwargs = {'url':url_, 'zip_dir':zip_dir_, 'unzipped_dir':unzipped_dir_},
    dag=dag)

t2 = PythonOperator(
    task_id='read',
    python_callable=read_df,
    provide_context = True,  
    op_kwargs = {'unzipped_dir':unzipped_dir_},
    dag=dag)

t3 = PythonOperator(
    task_id='feature_engineer',
    python_callable=feture_engineer,
    provide_context = True,
    dag=dag)

t4 = PythonOperator(
    task_id='train',
    python_callable = train_model,
    provide_context = True,  
    dag=dag)

t5 = PythonOperator(
    task_id='s3',
    python_callable=connect,
    dag=dag)




t5.set_upstream(t4)
t4.set_upstream(t3)
t3.set_upstream(t2)
t2.set_upstream(t1)
t1.set_upstream(t0)