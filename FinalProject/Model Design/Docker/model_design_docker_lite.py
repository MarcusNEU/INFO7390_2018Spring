from __future__ import division
import pandas as pd
from s3fs.core import S3FileSystem
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from model_design import ClassificationModel


S3 = S3FileSystem()
df = pd.DataFrame()
df_fe = pd.DataFrame()
X_test_bm = pd.DataFrame()
y_test_bm = pd.DataFrame()
X_test_fe = pd.DataFrame()
y_test_fe = pd.DataFrame()
lr = LogisticRegression(random_state=0)
rf = RandomForestClassifier(n_estimators=100, random_state=0)
bm_lr = ClassificationModel('Logistic Regression', lr, '', '', '', '')
bm_rf = ClassificationModel('Random Forest', rf, '', '', '', '')
fe_lr = ClassificationModel('Logistic Regression', lr, '', '', '', '')
fe_rf = ClassificationModel('Random Forest', rf, '', '', '', '')
optimal_lr = ClassificationModel('Logistic Regression', '', '', '', '', '')
optimal_rf = ClassificationModel('Random Forest', rf, '', '', '', '')


def read_dataset():
    try:
        print "==============================================================================="
        print "Connecting to S3..."
        print "Please make sure you have set your S3 access/secret access key in your system."
        global S3
        S3 = S3FileSystem(anon=False)
        print "Connected to S3!"
        print "==============================================================================="
    except:
        import traceback
        traceback.print_exc()
        print "Failed to connect to S3!"
        print "If you have not set your S3 access/secret key, please follow the instruction:"
        print "Make sure you have Python and pip in your system, and install aws cli:"
        print "Type in the following command in your terminal:"
        print "pip install --upgrade awscli"
        print "After aws cli installed, type in:"
        print "aws configure"
        print "Then you can set your access/secret key. Good luck!"
        print "==============================================================================="

    try:
        print "=============================================="
        print "Downloading the data set from Amazon S3..."
        BUCKET_NAME = 'info7390-2018spring-team2-final-dataset'
        DATASET = 'creditcard.csv'
        global df
        df = pd.read_csv(S3.open('{}/{}'.format(BUCKET_NAME, DATASET), mode='rb'))
        print df.info()
        print "Read data set successfully!"
        print "=============================================="
    except:
        import traceback
        traceback.print_exc()
        print "Failed to download or read the data set!"
        print "=============================================="


def data_preparation():
    print "=============================================="
    print "Preparing the data for model design..."
    global df
    global df_fe
    global X_test_bm
    global y_test_bm
    global X_test_fe
    global y_test_fe
    global bm_lr
    global bm_rf
    global fe_lr
    global fe_rf
    df_fe = df.drop(['V8', 'V13', 'V15', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'], axis=1)
    X_test_bm = df.iloc[:, df.columns != "Class"]
    y_test_bm = df.iloc[:, df.columns == "Class"]
    X_test_fe = df_fe.iloc[:, df_fe.columns != "Class"]
    y_test_fe = df_fe.iloc[:, df_fe.columns == "Class"]
    bm_lr = ClassificationModel('Logistic Regression', lr, df.columns.tolist(), '', '', '')
    fe_lr = ClassificationModel('Logistic Regression', lr, df_fe.columns.tolist(), '', '', '')
    bm_rf = ClassificationModel('Random Forest', rf, df.columns.tolist(), '', '', '')
    fe_rf = ClassificationModel('Random Forest', rf, df_fe.columns.tolist(), '', '', '')
    print "Done!"
    print "=============================================="


def benchmark():
    print "=============================================="
    print "1. Training benchmarks"
    print "=============================================="
    print "---------------------------------------------"
    print "Training Linear Regression benchmark..."
    print "---------------------------------------------"
    bm_lr.optimal_proportion(range(1, 3), df, X_test_bm, y_test_bm)
    print "---------------------------------------------"
    print "Training Random Forest benchmark..."
    print "---------------------------------------------"
    bm_rf.optimal_proportion(range(1, 3), df, X_test_bm, y_test_bm)
    print "---------------------------------------------"
    print "Benchmarks have been train!"
    print "---------------------------------------------"
    print ""


def feature_engineering():
    print "=============================================="
    print "2. Feature Engineering"
    print "=============================================="
    global optimal_lr
    global optimal_rf
    print "------------------------------------------------------------"
    print "Training Linear Regression after feature engineering..."
    print "------------------------------------------------------------"
    fe_lr.optimal_proportion(range(1, 3), df_fe, X_test_fe, y_test_fe)
    print "------------------------------------------------------------"
    print "Training Random Forest after feature engineering..."
    print "------------------------------------------------------------"
    fe_rf.optimal_proportion(range(1, 3), df_fe, X_test_fe, y_test_fe)
    if fe_lr < bm_lr:
        print "------------------------------------------------------------"
        print "Logistic Regression is improved after feature engineering!"
        print ""
        optimal_lr = fe_lr
    else:
        print "Logistic Regression is not improved after feature engineering!"
        print ""
        optimal_lr = bm_lr
    if fe_rf < bm_rf:
        print "Random Forest is improved after feature engineering!"
        optimal_rf = fe_rf
    else:
        print "Random Forest is not improved after feature engineering!"
        optimal_rf = bm_rf
    print "Feature engineering is done!"
    print "------------------------------------------------------------"


def feature_selection():
    print "=============================================="
    print "3. Feature Selection"
    print "=============================================="
    print "------------------------------------------------"
    print "feature selection for Logistic Regression..."
    print "------------------------------------------------"
    optimal_lr.feature_importance(df.loc[:, optimal_lr.columns], 0.75)
    print "------------------------------------------------"
    print "feature selection for Random Forest..."
    print "------------------------------------------------"
    optimal_rf.feature_importance(df.loc[:, optimal_rf.columns], 0.75)
    print "------------------------------------------------"
    print "Feature selection is done!"
    print "------------------------------------------------"


def model_selection():
    print "=============================================="
    print "4. Model Selection"
    print "=============================================="
    print "-----------------------------------------------------"
    print "tuning hyperparameters for Logistic Regression"
    print "-----------------------------------------------------"
    optimal_lr.lr_tuning_hyperparas(df.loc[:, optimal_lr.columns])
    optimal_lr.introduce()
    model_rank = sorted([optimal_lr, optimal_rf])
    print "-----------------------------------------------------"
    print "The model performances rank is:"
    print "1. ", model_rank[0].name
    print "2. ", model_rank[1].name
    print ""
    print "The best model is:"
    print model_rank[0].introduce()
    print "-----------------------------------------------------"


print "=============================================="
print "Credit Card Fraud Detection Model Design"
print "=============================================="
read_dataset()
data_preparation()
benchmark()
feature_engineering()
feature_selection()
model_selection()
print "=============================================="
print "Model design processing is done!"
print "=============================================="