## Web Application for Credit Card Frauds Detection

The application has been deployed on:

http://104.236.116.16:5000

This is the 2nd part of the final project in which the deployment of the classification website was designed. Flask and sqlite was used to establish the application. 

* ER picture for bank relationship database

A relational database is used to store relationships between tables and data. Three tables are established: users, posts, roles. The relationship between these three table is following. 
 
![Aaron Swartz](https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Web%20Application/graphs/Picture1.png)

* Flow chart for flask application

The application achieves the following basic functions: user login, user identification, Input data classification, File data classification. because of the confidentiality of the algorithm used to convert columns, classification of input data need to be further improved.

![Aaron Swartz](https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Web%20Application/graphs/Picture2.png)

* Docker

Use the following cmd to run the script:

docker pull xuchenlian/flask_final

docker run -p 5000:5000 xuchenlian/flask_final python app.py


* Try out the application!

 * Customer:
 
 User name: marcus
 
 Password: dog
 
 test data: https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Web%20Application/test_data_file/usercresdit.csv

 * Administrators:
 
 User name: john
 
 Password: cat
 
 test data:
 https://github.com/MarcusNEU/INFO7390_2018Spring/blob/master/FinalProject/Web%20Application/test_data_file/classification.csv

