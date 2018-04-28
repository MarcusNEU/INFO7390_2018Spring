## Background

The credit card data set comes from the European credit card tansaction data during two days in September 2013. In the total number of 284,807 times of transactions, 492 cases of fraud were discovered, which means the data set is extremely unbalanced, and fraud frequency only accounts for 0.172% of the total transactions.
<br>
This data set has 31 columns. V1, V2,,... ,V28 are the main features which are processed by PCA because of its sensibility, 'Time' is the time in seconds between each transaction and the first transaction. 'Amount' represents the transaction amount. 'Class' represents the response variable, 1 means fraud, and 0 means normal.

We will be using Logistic Regression, Support Vector Machine, Random Forest and Gradient Boosting Tree in the following steps.

![Aaron Swartz](https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Model%20Design/graphs/class_distribution.png)

## Precision-Recall Trade-off

To get a trade-off of precision and recall rate, we need to define a method to measure the overall performance of a model with a certain precision and recall rate. Instead of use the classification score in sklearn or some other packages, we need to make this have more reality meanings to customers, busniess and bank. In this project, we simply use the average amount of normal and fraud transactions to represent the weights of precision and recall.
  <pre><code>
  loss = (1 - precision) * 88.29 + (1 - recall) * 122.12
  </pre></code>


We tried total loss of FN and FP cases like loss = FN*122.12 + FP * 88.29 or divided by (FN+FP) which equals to the average loss of misclassification cases. These two are better than the former as they have clearer definitions and mathematical meanings. Actually, the current loss function we used is an approximate for average loss of frauds and misclassification transactions. It will get an approximate optimal solve under a smaller undersampling proportion compared with the other two loss functions. Thus, the one we used is the optimal one for calculation and research in a limited time.

## Benchmark

https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Model%20Design/benchmark.ipynb

In this part, two sets of model will be trained:

* Data set without resampling: all the models will be trianed on 75% of the original data set and test on the other 25%.
  <pre><code>
  X = df.iloc[:, df.columns != "Class"]
  y = df.iloc[:, df.columns == "Class"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
  </pre></code>
 
 The result is shown as follows:


|         Models         | Precision | Recall |  F1  | Accuracy | Custom Loss |
|:----------------------:|:---------:|:------:|:----:|:--------:|:-----------:|
|   Logistic Regression  |    0.81   |  0.60  | 0.69 |   0.99   |    65.62    |
| Support Vector Machine |    1.00   |  0.03  | 0.05 |   0.99   |    118.46   |
|      *Random Forest    |    0.94   |  0.75  | 0.83 |   0.99   |    35.83    | 
| Gradient Boosting Tree |    0.80   |  0.47  | 0.60 |   0.99   |    82.38   |

* Data set with resampling: all the models will be trained on 75% of the resampling data set and test on the whole data set.

	Here the optimal undersampling proportions for all the 4 models will be found out by iteration. Thus, after all the iteration, we will get the optimal proportions together with the min loss value of each model.

## Feature Engineering

https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Model%20Design/feature_engineering%26prediction_algorithms.ipynb

![Aaron Swartz](https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Model%20Design/graphs/distribution.png)

By looking into the differences between normal and fraud transactions, we get some insights of how to do feature engineering on anonymous features. Since this is a classification problem, two classes of the same feature share a very similar distribution would not do much to the model learning.

We simply drop all of the features that have very similar distributions between the two types of transactions.

  <pre><code>
  df = df.drop(['V8','V13','V15','V20','V22','V23','V24','V25','V26','V27','V28'], axis=1)
  </pre></code>
 
 Then the same things as we did for benchmark: iteration for optimal undersampling proportions. Finally we get results of optimal proportion and min loss of each model.
 
 We compare the performance of benchmark and the models after feature engineering. Then we pass the better one to do feature selection.

## Feature Selection

https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Model%20Design/feature_selection.ipynb

In this part, all the model will be trained on the corresponding optimal undersampling proportion calculated from the former parts. we simply use feature importance to rank the features and select top X features for all the models.

Since non-linear SVM is a black box classifier for which we do not know the mapping function Φ explicitly. There will be not any further pre-processing on SVM. The benchmark reaches the min loss at exactly the same proportion (=8) as the one after feature engineering. Thus, we will be using the one after feature engineering for further discussion.

Here is the result of the whole pre-procssing part:

* Logistic Regression:

  | Features          | Precision | Recall | F1   | Accuracy |
  |-------------------|-----------|--------|------|----------|
  | 19 selected by fe | 0.54      | 0.54   | 0.54 | 0.99     |
  | Top 21            | 0.34      | 0.66   | 0.47 | 0.99     |
  | All*              | 0.60      | 0.51   | 0.55 | 0.99     |
  
  Optimal proportion: 178
  
* SVM:

  | Features          | Precision | Recall | F1   | Accuracy |
  |-------------------|-----------|--------|------|----------|
  | 19 selected by fe*| 1.00      | 0.77   | 0.54 | 0.99     |
  | All               | 1.00      | 0.77   | 0.47 | 0.99     |
  
  Optimal proportion: 8

* Random Forest:

  | Features        | Precision | Recall | F1 Score |  Accuracy  |
  |-----------------|-----------|--------|----------|------------|
  | 19 picked by fe*| 0.82      | 0.96   | 0.88     | 0.99       |
  | Top 11          | 0.78      | 0.96   | 0.86     | 0.99       |
  | Top 6           | 0.79      | 0.96   | 0.87     | 0.99       |
  | All features    | 0.81      | 0.96   | 0.88     | 0.99       |
  
  Optimal proportion: 70
  
 * GBT:
 
 | Features        |Precision| Recal | F1 Score |Accuracy  | Custom Loss |
 |-----------------|---------|-------|----------|----------|-------------|
 | 19 picked by fe |0.74     | 0.88  | 0.81     | 0.99     | 37.61       |
 | Top 6 after fe* |0.68     | 0.93  | 0.79     | 0.99     | 36.80       |
      Optimal Proportion: 116
 
 
#### Here is the optimal models we got after pre-processing. They will be further dicussed in the next part.

 |         Models         | Precision | Recall |  F1  | Accuracy | Custom Loss | Diff vs 1st table |
|:----------------------:|:---------:|:------:|:----:|:--------:|:-----------:|:-----------------:|
|   Logistic Regression  |    0.60   |  0.51  | 0.55 |   0.99   |    95.15    |        29.53      |
| Support Vector Machine |    1.00   |  0.77  | 0.87 |   0.99   |    28.09    |       -90.37      |
|      *Random Forest    |    0.82   |  0.96  | 0.88 |   0.99   |    20.78    |       -15.05      |
| Gradient Boosting Tree |    0.68   |  0.93  | 0.79 |   0.99   |    36.80    |       -45.58      |
 

    
 ## Model Selection:
 
 https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Model%20Design/model_selection.ipynb
 
 This is the final part of the actual model design process. We will be tuning hyperparameters for all the current optimal models we got from the former parts. Grid search methods are defined here to implement the tuning.

Here are the optimal hyperparameters:
 
* Logistic Regression:

	* C=10 
	* penalty=l1

* SVM:

	* C=1
	* gamma=0.1
    
* Random Forest:

	* n_estimators=200
	* max_features=1
	* max_depth=20
	* min_samples_split=2
	* min_samples_leaf=1

* GBT:

	* leanring_rate=0.005
	* n_estimators=6000
	* max_depth=5
	* min_samples_split=48

Here is result of model selection:
    
|         Models         | Precision | Recall |  F1  | Accuracy | Custom Loss | Diff vs 1st table | Diff vs 2nd table |
|:----------------------:|:---------:|:------:|:----:|:--------:|:-----------:|:-----------------:|:-----------------:|
|   Logistic Regression  |    0.79   |  0.78  | 0.79 |   0.99   |    45.41    |       -20.21      |       -49.74      |
| *Support Vector Machine|    1.00   |  0.78  | 0.88 |   0.99   |    26.87    |       -91.59      |       -1.22       |
|      *Random Forest    |    0.86   |  0.94  | 0.90 |   0.99   |    19.69    |       -16.14      |       -1.09       |
| Gradient Boosting Tree |    0.80   |  0.96  | 0.87 |   0.99   |    22.54    |       -59.84      |       -14.26      |

* All the models have got improvements from tuning hyperparameters especially for Logistic Regression and Gradient Boosting Tree. In conclusion, SVM and Random Forest are more sensitive to the proportion of undersampling compared to tuning hyperparameters. Logistic Regression and GBT are sensitive to both of them.

* Random Forest is still the best among the 4 models. Actually the improvement after tuning hyperparameters is trivial for Support Vector Machine (non-linear) and Random Forest. Anyway, better than nothing. It still saves about $ 1.09 per (fraud + misclassification normal) transaction for us.

* SVM has a 100% precision which means if a transaction is classified as fraud by SVM then it must be a fraud. Thus, a typical predication process would be like this: predict using SVM first, if result is fraud then classify the transaction as fraud. Otherwise, predict again using Random Forest and return the prediction result. By doing this, we could increase the recall rate as far as possible without doing any harm to precision.

* The models we will be using in the web application are:
	* SVC(C=1, gamma=0.1)
	* RandomForestClassifier(n_estimators=200, max_features=1, max_depth=20, min_smaples_split=2, min_samples_leaf=1)

## Model & Columns Upload

https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Model%20Design/pickle_models.ipynb

SVM and Random Forest models have been pickled and uploaded to Amazon S3 for later using in web application:

![Aaron Swartz](https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Model%20Design/graphs/pickle.png)

## Dockerize

Use the following commands to pull and run the whole program:
Lite version with only Logistic Regression and Random Forest and less iteration:

docker pull qianli94neu/info7390final_lite

docker run -it qianli94neu/info7390final_lite python /src/model_design_docker_lite.py

Complete version:

docker pull qianli94neu/info7390final_complete

docker run -it qianli94neu/info7390final_complete python /src/model_design_docker.py

## References

We got inspirations on how to do feature engineering on anonymous features from:

https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow

We learned about how to iterate the undersampling proportion from:

https://www.kaggle.com/gargmanish/how-to-handle-imbalance-data-study-in-detail

We gained some insights of how to do grid search and cross validation on Logisitc Regression from:

https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now

We dicided which hyperparameters to tune by the guide of:

https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
