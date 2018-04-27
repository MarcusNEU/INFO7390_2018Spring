## background
The credit card data set comes from the European credit card tansaction data during two days in September 2013. In the total number of 284,807 times of transactions, 492 cases of fraud were discovered, which means the data set is extremely unbalanced, and fraud frequency only accounts for 0.172% of the total transactions.
<br>
This data set has 31 columns. V1, V2,,... ,V28 are the main features which are processed by PCA because of its sensibility, 'Time' is the time in seconds between each transaction and the first transaction. 'Amount' represents the transaction amount. 'Class' represents the response variable, 1 means fraud, and 0 means normal.

We will be using Logistic Regression, Support Vector Machine, Random Forest and Gradient Boosting Tree in the following steps.

## Benchmark
In this part, two sets of model will be trained:
> * Data set without resampling: all the models will be trianed on 75% of the original data set and test on the other 25%.
X = df.iloc[:, df.columns != "Class"]
y = df.iloc[:, df.columns == "Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
> * Data set with resampling: all the models will be trained on 75% of the resampling data set and test on the whole data set.
