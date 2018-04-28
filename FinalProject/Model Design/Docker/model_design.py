from __future__ import division
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold


class ClassificationModel:
    def __init__(self, name, model, columns, proportion, loss, report):
        self.name = name
        self.model = model
        self.columns = columns
        self.proportion = proportion
        self.loss = loss
        self.report = report

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        else:
            return self.loss == other.loss

    def __lt__(self, other):
        return self.loss < other.loss

    def introduce(self):
        print 'this is current optimal ' + self.name + ' model'
        print 'using columns: ', self.columns
        print 'using undersample proportion:', self.proportion
        print 'custom loss:', self.loss
        print 'classification report:'
        print self.report
        print 'model information:'
        print self.model

    @staticmethod
    def undersample(data, multiple):  # multiple denote the normal data = multiple * fraud data
        count_fraud_transaction = len(data[data["Class"] == 1])  # fraud by 1
        fraud_indices = np.array(data[data.Class == 1].index)
        normal_indices = np.array(data[data.Class == 0].index)
        normal_indices_undersample = np.array(
            np.random.choice(normal_indices, (multiple * count_fraud_transaction), replace=False))
        undersample_data = np.concatenate([fraud_indices, normal_indices_undersample])
        undersample_data = data.iloc[undersample_data, :]
        return undersample_data

    @staticmethod
    def custom_loss_function(model, features_train, features_test, labels_train, labels_test):
        model.fit(features_train, labels_train.values.ravel())
        pred = model.predict(features_test)
        cm = confusion_matrix(labels_test, pred)
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        loss = (1 - precision) * 88.29 + (1 - recall) * 122.12
        return loss

    def prediction_algorithms(self, features_train, features_test, labels_train, labels_test):
        self.model.fit(features_train, labels_train.values.ravel())
        pred = self.model.predict(features_test)
        cm = confusion_matrix(labels_test, pred)
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        self.loss = (1 - precision) * 88.29 + (1 - recall) * 122.12
        print "the recall for this model is :", recall
        print "the precision for this model is :", precision
        print "the custom loss is:", self.loss
        print "The accuracy is :", (cm[1, 1]+cm[0, 0])/(cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
        print "TP", cm[1, 1] # no of fraud transaction which are predicted fraud
        print "TN", cm[0, 0] # no of normal transaction which are predicted normal
        print "FP", cm[0, 1] # no of normal transaction which are predicted fraud
        print "FN", cm[1, 0] # no of fraud Transaction which are predicted normal
        self.report = classification_report(labels_test, pred)
        print "Classification Report:"
        print self.report
        # return classification_report(labels_test, pred), loss, model
        # return loss

    def optimal_proportion(self, proportion_range, data, features_test, labels_test):
        proportion = []
        loss_list = []
        # for i in range(1, 5):
        for i in proportion_range:
            undersample_data = self.undersample(data, i)
            X_undersample = undersample_data.iloc[:, undersample_data.columns != "Class"]
            y_undersample = undersample_data.iloc[:, undersample_data.columns == "Class"]
            X_undersample_train, X_undersample_test, y_undersample_train, y_undersample_test = train_test_split(
                X_undersample, y_undersample, random_state=0)
            loss_list.append(self.custom_loss_function(self.model, X_undersample_train, features_test,
                                                       y_undersample_train, labels_test))
            proportion.append(i)
        min_loss = np.min(loss_list)
        min_loss_index = loss_list.index(min_loss)
        optimal_proportion = proportion[min_loss_index]

        print "optimal proportion of normal/fraud is : ", optimal_proportion
        self.proportion = optimal_proportion
        undersample_data = self.undersample(data, optimal_proportion)
        print ""
        print "the model classification for {} proportion".format(optimal_proportion)
        X_undersample = undersample_data.iloc[:, undersample_data.columns != "Class"]
        y_undersample = undersample_data.iloc[:, undersample_data.columns == "Class"]
        X_undersample_train, X_undersample_test, y_undersample_train, y_undersample_test = train_test_split(X_undersample,
                                                                                                            y_undersample,
                                                                                                            random_state=0)
        self.prediction_algorithms(X_undersample_train, features_test, y_undersample_train, labels_test)
        print "________________________________________________________________________________________________________"
        print ""

    def feature_importance(self, df, toprate):
        importances = ''
        if self.name == "Logistic Regression":
            print "calculating best features for prediction of current optimal Logistic Regression model..."
            importances = self.model.coef_[0]
        elif self.name == "Random Forest" or self.name == "Gradient Boosting Tree":
            print "calculating best features for prediction of current optimal ", self.name, " model..."
            importances = self.model.feature_importances_
        data = df.loc[:, self.columns]
        undersample_data = self.undersample(data, self.proportion)
        X_undersample = undersample_data.iloc[:, undersample_data.columns != "Class"]
        y_undersample = undersample_data.iloc[:, undersample_data.columns == "Class"]
        X_undersample_train, X_undersample_test, y_undersample_train, y_undersample_test = train_test_split(
            X_undersample, y_undersample, random_state=0)
        self.model.fit(X_undersample_train / np.std(X_undersample_train, 0), y_undersample_train.values.ravel())
        feature_importances = [(feature, abs(round(importance, 2))) for feature, importance in
                                  zip(X_undersample_train.columns, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        print feature_importances
        features_ranked = []
        for i in range(0, len(feature_importances)):
            features_ranked.append(feature_importances[i][0])

        loss = []
        features_list = []
        for i in range(0, int(len(features_ranked)*toprate)):
            print "using the top ", i+1, " features for prediction..."
            features = features_ranked[0:i+1]
            features.append('Class')
            data_ = data.loc[:, features]
            undersample_data = self.undersample(data_, self.proportion)
            X_undersample = undersample_data.iloc[:, undersample_data.columns != "Class"]
            y_undersample = undersample_data.iloc[:, undersample_data.columns == "Class"]
            X_undersample_train, X_undersample_test, y_undersample_train, y_undersample_test = train_test_split(
                X_undersample, y_undersample, random_state=0)
            X_test = data_.iloc[:, data_.columns != "Class"]
            y_test = data_.iloc[:, data_.columns == "Class"]
            loss.append(self.custom_loss_function(self.model, X_undersample_train, X_test, y_undersample_train, y_test))
            features_list.append(features)
        min_loss = np.min(loss)
        min_loss_index = loss.index(min_loss)
        optimal_features = features_list[min_loss_index]

        if min_loss < self.loss:
            print "****************************************************************************"
            print self.name, "is improved after feature selection! Model parameters updated!"
            print "Using top ", len(optimal_features)-1, " features gives the best result for ", self.name
            print "****************************************************************************"
            self.columns = optimal_features
            optimal_data = df.loc[:, self.columns]
            undersample_data = self.undersample(optimal_data, self.proportion)
            X_undersample = undersample_data.iloc[:, undersample_data.columns != "Class"]
            y_undersample = undersample_data.iloc[:, undersample_data.columns == "Class"]
            X_undersample_train, X_undersample_test, y_undersample_train, y_undersample_test = train_test_split(
                X_undersample, y_undersample, random_state=0)
            X_test = optimal_data.iloc[:, optimal_data.columns != "Class"]
            y_test = optimal_data.iloc[:, optimal_data.columns == "Class"]
            self.prediction_algorithms(X_undersample_train, X_test, y_undersample_train, y_test)
        else:
            print self.name, "is not improved after feature selection! Model parameters remained."

    def lr_tuning_hyperparas(self, df):
        if self.name == "Logistic Regression":
            print "------------------------------------------------"
            print "tuning C and penalty..."
            print "------------------------------------------------"
            undersample_data_lr = self.undersample(df, self.proportion)
            X_undersample_lr = undersample_data_lr.iloc[:, undersample_data_lr.columns != "Class"]
            y_undersample_lr = undersample_data_lr.iloc[:, undersample_data_lr.columns == "Class"]
            X_undersample_train_lr, X_undersample_test_lr, y_undersample_train_lr, y_undersample_test_lr = \
                train_test_split(X_undersample_lr, y_undersample_lr, random_state=0)
            X_test_lr = df.iloc[:, df.columns != "Class"]
            y_test_lr = df.iloc[:, df.columns == "Class"]

            fold = KFold(n_splits=5, shuffle=False, random_state=0)

            c_param_range = [0.01, 0.1, 1, 10, 100]
            penalties = ['l1', 'l2']

            penalty_list = []
            c_list = []
            mean_loss_list = []

            for penalty in penalties:
                print '-------------------------------------------'
                print 'Penalty: ', penalty
                print '-------------------------------------------'
                for c_param in c_param_range:
                    print '-------------------------------------------'
                    print 'C parameter: ', c_param
                    print '-------------------------------------------'
                    print ''

                    loss_list = []
                    for k, (train, test) in enumerate(fold.split(X_undersample_train_lr, y_undersample_train_lr)):
                        # Call the logistic regression model with a certain C parameter
                        lr_tuning = LogisticRegression(C=c_param, penalty=penalty, random_state=0)

                        # Calculate the custom loss and append it to a list for loss representing the current c_parameter
                        loss = self.custom_loss_function(lr_tuning, X_undersample_train_lr.iloc[train], X_test_lr,
                                                         y_undersample_train_lr.iloc[train], y_test_lr)
                        loss_list.append(loss)
                        print 'Fold ', k + 1, ': loss = ', loss

                    print ''
                    print 'Mean loss', np.mean(loss_list)
                    print ''
                    penalty_list.append(penalty)
                    c_list.append(c_param)
                    mean_loss_list.append(np.mean(loss_list))
            results_table = pd.DataFrame(index=range(len(mean_loss_list)), columns=['Penalty', 'C_parameter', 'Mean loss'])
            results_table['Penalty'] = penalty_list
            results_table['C_parameter'] = c_list
            results_table['Mean loss'] = mean_loss_list

            best_penalty = results_table.loc[results_table['Mean loss'].idxmin()]['Penalty']
            best_c = results_table.loc[results_table['Mean loss'].idxmin()]['C_parameter']

            print results_table
            print ''

            # Finally, we can check which C parameter is the best amongst the chosen.
            print '************************************************************************************'
            print 'Best model to choose from cross validation is with Penalty = ', best_penalty, 'and best c = ', best_c
            print '************************************************************************************'

            self.model = LogisticRegression(C=best_c, penalty=best_penalty, random_state=0)
            self.prediction_algorithms(X_undersample_train_lr, X_test_lr, y_undersample_train_lr, y_test_lr)
        else:
            print "This method can only be called by Logistic Regression!"

    def svm_tuning_hyperparas(self, df):
        if self.name == "Support Vector Machine":
            print "------------------------------------------------"
            print "tuning C and gamma..."
            print "------------------------------------------------"
            undersample_data_svm = self.undersample(df, self.proportion)
            X_undersample_svm = undersample_data_svm.iloc[:, undersample_data_svm.columns != "Class"]
            y_undersample_svm = undersample_data_svm.iloc[:, undersample_data_svm.columns == "Class"]
            X_undersample_train_svm, X_undersample_test_svm, y_undersample_train_svm, y_undersample_test_svm = train_test_split(
                X_undersample_svm, y_undersample_svm, random_state=0)
            X_test_svm = df.iloc[:, df.columns != "Class"]
            y_test_svm = df.iloc[:, df.columns == "Class"]
            fold = KFold(n_splits=5, shuffle=False, random_state=0)

            c_param_range = [0.1, 1, 2, 5]
            gamma_range = [0.01, 0.1, 'auto', 1]

            c_list = []
            gamma_list = []
            mean_loss_list = []

            for c_param in c_param_range:
                print "-------------------------------------------"
                print "C parameter: ", c_param
                print "-------------------------------------------"
                for gamma in gamma_range:
                    print '-------------------------------------------'
                    print 'Gamma: ', gamma
                    print '-------------------------------------------'
                    print ""

                    loss_list = []
                    for k, (train, test) in enumerate(fold.split(X_undersample_train_svm, y_undersample_train_svm)):
                        # Call the logistic regression model with a certain C parameter
                        svm_tuning = SVC(C=c_param, gamma=gamma, random_state=0)

                        # Calculate the custom loss and append it to a list for loss representing the current c_parameter
                        loss = self.custom_loss_function(svm_tuning, X_undersample_train_svm.iloc[train], X_test_svm,
                                                         y_undersample_train_svm.iloc[train], y_test_svm)
                        loss_list.append(loss)
                        print 'Fold ', k + 1, ': loss = ', loss

                    print ''
                    print 'Mean loss', np.mean(loss_list)
                    print ''
                    gamma_list.append(gamma)
                    c_list.append(c_param)
                    mean_loss_list.append(np.mean(loss_list))
            results_table = pd.DataFrame(index=range(len(mean_loss_list)), columns=['C_parameter', 'Gamma', 'Mean loss'])
            results_table['Gamma'] = gamma_list
            results_table['C_parameter'] = c_list
            results_table['Mean loss'] = mean_loss_list

            best_gamma = results_table.loc[results_table['Mean loss'].idxmin()]['Gamma']
            best_c = results_table.loc[results_table['Mean loss'].idxmin()]['C_parameter']

            print results_table
            print ""

            # Finally, we can check which C parameter is the best amongst the chosen.
            print '************************************************************************************'
            print 'Best model to choose from cross validation is with C = ', best_c, 'and best gamma = ', best_gamma
            print '************************************************************************************'

            self.model = SVC(C=best_c, gamma=best_gamma, random_state=0)
            self.prediction_algorithms(X_undersample_train_svm, X_test_svm, y_undersample_train_svm, y_test_svm)
        else:
            print "This method can only be called by Support Vector Machine!"

    def rf_tuning_hyperparas(self, df):
        if self.name == "Random Forest":
            print "------------------------------------------------"
            print "tuning n_estimators, max_features, max_depth..."
            print "------------------------------------------------"
            undersample_data_rf = self.undersample(df, self.proportion)
            X_undersample_rf = undersample_data_rf.iloc[:, undersample_data_rf.columns != "Class"]
            y_undersample_rf = undersample_data_rf.iloc[:, undersample_data_rf.columns == "Class"]
            X_undersample_train_rf, X_undersample_test_rf, y_undersample_train_rf, y_undersample_test_rf = train_test_split(
                X_undersample_rf, y_undersample_rf, random_state=0)
            X_test_rf = df.iloc[:, df.columns != "Class"]
            y_test_rf = df.iloc[:, df.columns == "Class"]

            fold = KFold(n_splits=5, shuffle=False, random_state=0)

            n_estimators_range = [10, 100, 150, 200]
            max_features_type = ['auto', 'log2', 1, 2, 8]
            max_depth_range = [10, 20, 30, None]
            n_estimators_list = []
            max_features_list = []
            max_depth_list = []
            mean_loss_list = []

            for n_estimators in n_estimators_range:
                print '-------------------------------------------'
                print 'n_estimators: ', n_estimators
                print '-------------------------------------------'
                for max_features in max_features_type:
                    print '-------------------------------------------'
                    print 'max_features: ', max_features
                    print '-------------------------------------------'
                    for max_depth in max_depth_range:
                        print '-------------------------------------------'
                        print 'max_depth: ', max_depth
                        print '-------------------------------------------'

                        loss_list = []
                        for k, (train, test) in enumerate(fold.split(X_undersample_train_rf, y_undersample_train_rf)):
                            tuning_rf1 = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                                        max_depth=max_depth, random_state=0)

                            loss = self.custom_loss_function(tuning_rf1, X_undersample_train_rf.iloc[train], X_test_rf,
                                                             y_undersample_train_rf.iloc[train], y_test_rf)
                            loss_list.append(loss)
                            print 'Fold ', k + 1, ': loss = ', loss

                        print ''
                        print 'Mean loss', np.mean(loss_list)
                        print ''
                        n_estimators_list.append(n_estimators)
                        max_features_list.append(max_features)
                        max_depth_list.append(max_depth)
                        mean_loss_list.append(np.mean(loss_list))
            results_table = pd.DataFrame(index=range(len(mean_loss_list)),
                                         columns=['n_estimators', 'max_features', 'max_depth', 'Mean loss'])
            results_table['n_estimators'] = n_estimators_list
            results_table['max_features'] = max_features_list
            results_table['max_depth'] = max_depth_list
            results_table['Mean loss'] = mean_loss_list

            best_n_estimators = results_table.loc[results_table['Mean loss'].idxmin()]['n_estimators']
            best_max_features = results_table.loc[results_table['Mean loss'].idxmin()]['max_features']
            best_max_depth = results_table.loc[results_table['Mean loss'].idxmin()]['max_depth']

            print results_table
            print ""

            print '**************************************************************************************************'
            print "Best model to choose from cross validation is with n_estimators = ", best_n_estimators, ", best max_features = ", best_max_features
            print "and best max_depth = ", best_max_depth
            print '**************************************************************************************************'

            print "------------------------------------------------"
            print "tuning min_samples_split, min_samples_leaf..."
            print "------------------------------------------------"

            min_samples_split_range = [2, 4, 8]
            min_samples_leaf_range = [1, 2, 4]

            min_samples_split_list = []
            min_samples_leaf_list = []
            mean_loss_list2 = []

            for min_samples_split in min_samples_split_range:
                print '-------------------------------------------'
                print 'min_samples_split: ', min_samples_split
                print '-------------------------------------------'
                for min_samples_leaf in min_samples_leaf_range:
                    print '-------------------------------------------'
                    print 'min_samples_leaf: ', min_samples_leaf
                    print '-------------------------------------------'

                    loss_list2 = []
                    for k, (train, test) in enumerate(fold.split(X_undersample_train_rf, y_undersample_train_rf)):
                        tuning_rf2 = RandomForestClassifier(n_estimators=best_n_estimators, max_features=best_max_features,
                                                    min_samples_split=min_samples_split,
                                                    min_samples_leaf=min_samples_leaf, max_depth=best_max_depth,
                                                    random_state=0)

                        loss2 = self.custom_loss_function(tuning_rf2, X_undersample_train_rf.iloc[train], X_test_rf,
                                                         y_undersample_train_rf.iloc[train], y_test_rf)
                        loss_list2.append(loss2)
                        print 'Fold ', k + 1, ': loss = ', loss2

                    print ''
                    print 'Mean loss', np.mean(loss_list2)
                    print ''
                    min_samples_split_list.append(min_samples_split)
                    min_samples_leaf_list.append(min_samples_leaf)
                    mean_loss_list2.append(np.mean(loss_list2))
            results_table2 = pd.DataFrame(index=range(len(mean_loss_list2)),
                                         columns=['min_samples_split', 'min_samples_leaf', 'Mean loss'])
            results_table2['min_samples_split'] = min_samples_split_list
            results_table2['min_samples_leaf'] = min_samples_leaf_list
            results_table2['Mean loss'] = mean_loss_list2

            best_min_samples_split = int(results_table2.loc[results_table2['Mean loss'].idxmin()]['min_samples_split'])
            best_min_samples_leaf = int(results_table2.loc[results_table2['Mean loss'].idxmin()]['min_samples_leaf'])

            print results_table2
            print ""

            print '********************************************************************************'
            print "Best model to choose from cross validation is with best min_samples_split = ", best_min_samples_split
            print "and best min_samples_leaf = ", best_min_samples_leaf
            print '********************************************************************************'

            self.model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth,
                                                max_features=best_max_features, min_samples_leaf=best_min_samples_leaf,
                                                min_samples_split=best_min_samples_split, random_state=0)
            self.prediction_algorithms(X_undersample_train_rf, X_test_rf, y_undersample_train_rf, y_test_rf)
        else:
            print "This method can only be called by Random Forest!"

    def gbt_tuning_hyperparas(self, df):
        if self.name == "Gradient Boosting Tree":
            print "-----------------------------------------------------"
            print "tuning n_estimators, max_depth, min_samples_split..."
            print "-----------------------------------------------------"
            undersample_data_gbt = self.undersample(df, self.proportion)
            X_undersample_gbt = undersample_data_gbt.iloc[:, undersample_data_gbt.columns != "Class"]
            y_undersample_gbt = undersample_data_gbt.iloc[:, undersample_data_gbt.columns == "Class"]
            X_undersample_train_gbt, X_undersample_test_gbt, y_undersample_train_gbt, y_undersample_test_gbt = train_test_split(
                X_undersample_gbt, y_undersample_gbt, random_state=0)
            X_test_gbt = df.iloc[:, df.columns != "Class"]
            y_test_gbt = df.iloc[:, df.columns == "Class"]

            fold = KFold(n_splits=5, shuffle=False, random_state=0)

            n_estimators_range = [100, 200, 300]
            max_depth_range = [3, 5, 7, 9]
            min_samples_split_range = [2, 4, 8, 16, 32, 48, 64, 80]
            n_estimators_list = []
            max_depth_list = []
            min_samples_split_list = []
            mean_loss_list = []

            for n_estimators in n_estimators_range:
                print '-------------------------------------------'
                print 'n_estimators: ', n_estimators
                print '-------------------------------------------'
                for max_depth in max_depth_range:
                    print '-------------------------------------------'
                    print 'max_depth: ', max_depth
                    print '-------------------------------------------'
                    for min_samples_split in min_samples_split_range:
                        print '-------------------------------------------'
                        print 'min_samples_split: ', min_samples_split
                        print '-------------------------------------------'

                        loss_list = []
                        for k, (train, test) in enumerate(fold.split(X_undersample_train_gbt, y_undersample_train_gbt)):
                            tuning_gbt = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                             min_samples_split=min_samples_split, random_state=0)

                            loss = self.custom_loss_function(tuning_gbt, X_undersample_train_gbt.iloc[train], X_test_gbt,
                                                             y_undersample_train_gbt.iloc[train], y_test_gbt)
                            loss_list.append(loss)
                            print 'Fold ', k + 1, ': loss = ', loss

                        print ''
                        print 'Mean loss', np.mean(loss_list)
                        print ''
                        n_estimators_list.append(n_estimators)
                        max_depth_list.append(max_depth)
                        min_samples_split_list.append(min_samples_split)
                        mean_loss_list.append(np.mean(loss_list))
            results_table = pd.DataFrame(index=range(len(mean_loss_list)),
                                         columns=['n_estimators', 'max_depth', 'min_samples_split', 'Mean loss'])
            results_table['n_estimators'] = n_estimators_list
            results_table['max_depth'] = max_depth_list
            results_table['min_samples_split'] = min_samples_split_list
            results_table['Mean loss'] = mean_loss_list

            best_n_estimators = int(results_table.loc[results_table['Mean loss'].idxmin()]['n_estimators'])
            best_max_depth = int(results_table.loc[results_table['Mean loss'].idxmin()]['max_depth'])
            best_min_samples_split = int(results_table.loc[results_table['Mean loss'].idxmin()]['min_samples_split'])

            print results_table
            print ""
            print '**************************************************************************************************'
            print "Best model to choose from cross validation is with n_estimators = ", best_n_estimators
            print "best max_depth = ", best_max_depth, "and best min_samples_split = ", best_min_samples_split
            print '**************************************************************************************************'

            print "------------------------------------------------"
            print "tuning subsample..."
            print "------------------------------------------------"

            subsample_range = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1]
            subsample_list = []
            mean_loss_list2 = []

            for subsample in subsample_range:
                print '-------------------------------------------'
                print 'subsample: ', subsample
                print '-------------------------------------------'
                loss_list2 = []
                for k, (train, test) in enumerate(fold.split(X_undersample_train_gbt, y_undersample_train_gbt)):
                    tuning_gbt2 = GradientBoostingClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth,
                                                             min_samples_split=best_min_samples_split, subsample=subsample,
                                                             random_state=0)

                    loss2 = self.custom_loss_function(tuning_gbt2, X_undersample_train_gbt.iloc[train], X_test_gbt,
                                                      y_undersample_train_gbt.iloc[train], y_test_gbt)
                    loss_list2.append(loss2)
                    print 'Fold ', k + 1, ': loss = ', loss2

                print ''
                print 'Mean loss', np.mean(loss_list2)
                print ''
                subsample_list.append(subsample)
                mean_loss_list2.append(np.mean(loss_list2))
            results_table2 = pd.DataFrame(index=range(len(mean_loss_list2)), columns=['Subsample', 'Mean loss'])
            results_table2['Subsample'] = subsample_list
            results_table2['Mean loss'] = mean_loss_list2

            best_subsample = results_table2.loc[results_table2['Mean loss'].idxmin()]['Subsample']

            print results_table2
            print ""

            print '*******************************************************************'
            print "Best model to choose from cross validation is with subsample = ", best_subsample
            print '*******************************************************************'

            print "------------------------------------------------"
            print "tuning learning_rate, n_estimators..."
            print "------------------------------------------------"

            learning_rate_range = [0.1, 0.05, 0.01, 0.005]
            n_estimators_range2 = [300, 600, 3000, 6000]
            mean_loss_list3 = []

            for i in range(0, 4):
                print '-------------------------------------------'
                print 'learning_rate: ', learning_rate_range[i]
                print 'n_estimators: ', n_estimators_range2[i]
                print '-------------------------------------------'
                loss_list3 = []
                for k, (train, test) in enumerate(fold.split(X_undersample_train_gbt, y_undersample_train_gbt)):
                    tuning_gbt3 = GradientBoostingClassifier(learning_rate=learning_rate_range[i],
                                                             n_estimators=n_estimators_range2[i],
                                                             max_depth=best_max_depth,
                                                             min_samples_split=best_min_samples_split,
                                                             subsample=best_subsample, random_state=0)

                    loss3 = self.custom_loss_function(tuning_gbt3, X_undersample_train_gbt.iloc[train], X_test_gbt,
                                                      y_undersample_train_gbt.iloc[train], y_test_gbt)
                    loss_list3.append(loss3)
                    print 'Fold ', k + 1, ': loss = ', loss3

                print ''
                print 'Mean loss', np.mean(loss_list3)
                print ''
                mean_loss_list3.append(np.mean(loss_list3))
            results_table3 = pd.DataFrame(index=range(len(mean_loss_list3)),
                                         columns=['Learning_rate', 'N_estimators', 'Mean loss'])
            results_table3['Learning_rate'] = learning_rate_range
            results_table3['N_estimators'] = n_estimators_range2
            results_table3['Mean loss'] = mean_loss_list3

            best_learning_rate = results_table3.loc[results_table3['Mean loss'].idxmin()]['Learning_rate']
            best_n_estimators = int(results_table3.loc[results_table3['Mean loss'].idxmin()]['N_estimators'])

            print results_table3
            print ""

            print "***************************************************************************"
            print "Best model to choose from cross validation is with best learning rate = ", best_learning_rate,
            print "and corresponding n_estimators = ", best_n_estimators
            print "***************************************************************************"

            self.model = GradientBoostingClassifier(learning_rate=best_learning_rate, n_estimators=best_n_estimators,
                                                    max_depth=best_max_depth, min_samples_split=best_min_samples_split,
                                                    random_state=0)
            self.prediction_algorithms(X_undersample_train_gbt, X_test_gbt, y_undersample_train_gbt, y_test_gbt)
        else:
            print "This method can only be called by Gradient Boosting Tree!"


