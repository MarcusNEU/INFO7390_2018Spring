## Credit Card Frauds Detection

#### This is the final project for INFO7390. The topic is credit card frauds detection which is a classic and also a hot field of data science.

#### Our goals of this project are shown below:

1. Accurately predict credit card fraud cases using imbalanced data 
2. To protect card holders' and commercial banks' benefits
3. Review and compare different fraud detection techniques and select the best model by trade-offs of different evaluation methods

#### Data Source:

https://www.kaggle.com/mlg-ulb/creditcardfraud
Anonymized credit card transactions labeled as fraudulent or genuine
The datasets contain transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

#### Progress Report:
---
#### Fisrt Week (Apr 7th - Apr 14th):
---
##### 1. Paper and code reviwe on how to do feature engineering on anonymous features:

Since 28 of 30 columns in this dataset are anonymous except the amount and class (fraud or not) of transactions. There is not many things we could do at the first look. Then we look into some other guy's researches and we found an interesting approach of feature engineering on this specific dataset: https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow.
In this work, the author compare the distribution of fraud transactions and normal transactions on every column. This is a classification problem, a good feature means it can tell the different between the possitive and negative class. Thus, if the distribution of two classes are very similiar for the same column, it is probably saying that this column does not have many things to do with the classification and it will be removed in the feature engineering.

We will keep on exploring other works about this dataset to get more ideas of feature engineering these two days and start our own job after that.

##### 2. Paper Review on how to deal with unbalanced data:

Basically we use resampling, and we might want to use undersampling in this specific case because our postive class takes only 0.172%. We found that different percentage of undersample will give us different model performances.Thus, we might wanna try out different portion of undersample on each model we will be using in our work to find the best portions for classification.

##### 3. Learning about the models and error metrics we want to used in this project:

As you can see in the project proposal, we want to use Biased penalty SVM, Logistic Regression, Random Forest, Gradient Boosting Tree in our work. There are acutally a lot of former reseaches on credit card fraud detection using neural network what we might not want to do the repetitive work here anyomre. Logistic Regression and Random Forest are also widely used in other works on this dataset, which could be good references for us to tunning the parameters and hyperparameters. Biased penalty SVM is designed for handling unbalanced data beacuse itself will oversample the trainset data. A lot of work includes decision tree but none of them using GBT as far as we found. These two models (BPSVM and GBT) are actually recommended by one of my friends who majored in data science which after our discussion which might be a good fit for this problem.

R2, Accuracy Rate, Recall Rate, Precision, AUC are considered to be used in our work because multiple metrics need to be used to measure the performance of classification problem on such a umblanced dataset. We found that there is actually a recall rate and precision trade-off problem in this case because the positive class is too small and we do not want miss any one of it which will unaviodablly descrease the precision of our prediction. 

---
#### Plan for 2nd week (Apr 14th - Apr 21st):
---
##### 1. Data Cleaning:

No missing value. Might want to try out remove outliers if there is any.

##### 2. EDA:

Histogram, distributions, pair plots, etc.

##### 3. Feature Engineering:

Refer to https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow, we will try to remove the columns where two classes have a very similar distribution and consider apply some other transfermations on the features.

##### 4. Divide Train/Test Dataset 

Applying resampling with different portions and then do train/test set split.

##### 5. Model Fitting:

Try out Biased penalty SVM on the whole dataset. Try out Logistic Regression, Random Forest and Gradient Boosting Tree on different portions of resampled dataset as well as the whole dataset as a benchmark. Also fit the model on the dataset without feature engineering to see whether the feature engineering is helping.

##### 6. Feature Selection:

Try  different approaches on feature selection like boruta, RFE, forward search on each model campared to the benchmark which using all the features.

---
#### Plan for 3rd week (Apr 21st - Apr 27th):
---
##### 1. Model Selection:

Tuning parameters and hyperparameters of each model. Using comprehensive error metrics to measure the performance of each model.

##### 2. Pipeline Design:

Might want to use one of Luigi/Airflow/Sklearn.

##### 3. Appilication Design:

Might want to design a web application with flask.

##### 4. Application Deployment:

Might want to upload the feature engineering, model and error metrics to the cloud and deploy the web application on the server.

###### *5. Real Time Data Processing and Online Database Deployment:

If we have enough time, we might want to try out Amazon Lambda to update the model implementing real time data processing. We might also want to consider database deployment which used for store the data uploaded to the application with primary constrains to avoid storing duplicate data.



