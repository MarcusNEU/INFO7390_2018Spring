import math
import pandas as pd
from sklearn.metrics import *
import numpy as np


class ErrorMetrics:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.error_metric = pd.DataFrame({'rmse_train': [],
                                          'rmse_test': [],
                                          'mae_train': [],
                                          'mae_test': [],
                                          'mape_train': [],
                                          'mape_test': [],
                                          'r_train': [],
                                          'r_test': []})

    def cal_metric(self, modelname, model):
        y_train_pre = model.predict(self.X_train)
        y_test_pre = model.predict(self.X_test)

        rmse_train = math.sqrt(mean_squared_error(self.y_train, y_train_pre))
        rmse_test = math.sqrt(mean_squared_error(self.y_test, y_test_pre))

        mae_train = mean_absolute_error(self.y_train, y_train_pre)
        mae_test = mean_absolute_error(self.y_test, y_test_pre)

        mape_train = np.mean(np.abs((self.y_train - y_train_pre) / self.y_train)) * 100
        mape_test = np.mean(np.abs((self.y_test - y_test_pre) / self.y_test)) * 100

        r_train = r2_score(self.y_train, y_train_pre)
        r_test = r2_score(self.y_test, y_test_pre)

        error_metric_local = pd.DataFrame({'Model': [modelname],
                                           'rmse_train': [rmse_train],
                                           'rmse_test': [rmse_test],
                                           'mae_train': [mae_train],
                                           'mae_test': [mae_test],
                                           'mape_train': [mape_train],
                                           'mape_test': [mape_test],
                                           'r_train': [r_train],
                                           'r_test': [r_test]})

        if self.error_metric.columns[0] == 'Model' and not self.error_metric.loc[self.error_metric['Model'] == modelname].empty:
            self.error_metric = self.error_metric[self.error_metric.Model != modelname]
        else:
            pass

        self.error_metric = pd.concat([self.error_metric, error_metric_local])