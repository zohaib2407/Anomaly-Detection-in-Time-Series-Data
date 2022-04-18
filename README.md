# Anomaly-Detection-in-Time-Series-Data
Goal is to detect anomaly in a time series data in manufacturing industry. Manufacturing industry is a heavy industry which requires large amount of capital investment on heavy machinery assets which are most critical for manufacturing. The ability to detect any anomaly in advance would result in mitigating the risk of an equipment failure. 

## Related Work/Current Industry Applications
Anomaly detection is a great application of machine learning. Few of the most widely used applications of anomaly detection include fraud detection, cyber-attack detection, and equipment monitoring. It is also commonly used to detect intrusions to a computer network and to detect the risk of medical problems in health data. Related work includes statistical frameworks for detecting Latent faults - Performance anomalies, that indicate a fault, or that could eventually result in a fault.

## Application of Machine Learning and Statistics 
There are many ways in which statistics and machine learning techniques can be leveraged for anomaly detection. We have used a few of them as part of our project and presented a comprehensive comparison between them in Python using Scikit-Learn. In our project we have explored both supervised and unsupervised  techniques for anomaly detection. Specifically, we have implemented the below mentioned techniques:
•	Interquartile range 
•	K means Clustering
•	Gaussian Distribution
•	Gaussian Mixture Model
For evaluation of these techniques, we will use different classifier evaluation metrics such as Number of Outliers detected, Confusion Matrix, Accuracy, etc. 

## Understanding the Dataset and Dataset Ingestion
Our dataset contains sensor readings from 52 sensors installed on a pump which is a part of a manufacturing setup. These different sensors measure various behaviours of the pump. The dataset contains reading for one full year at different timestamps. We also have a ‘machine status’ column which represents different working conditions of the pump as ‘normal operating ‘broken’, and ‘recovering. Our dataset has 220k different data points each of which represents a reading of the 53 different sensors at a given timestamp. After downloading the dataset, we ingested it into our project using a Pandas dataframe. 

## Data Science Pipeline

1.	Data Cleaning and pre-processing
The dataset we used had a few abnormalities, like empty columns, repetitive entries, missing values, and more. So, we had to clean the dataset. We have also utilized an interesting concept/technique called pickling for this project. Pickling is basically a technique of converting a python object into a byte stream to be stored in the file and maintain program state across all sessions in the project. Our cleaned dataset has a shape of (219521, 52).
 
2.	Exploratory Data Analysis
Below we visualize the distribution of some of the sensors.
We can see most of the sensor readings follow a normal distribution which is intuitive and this observation lets us use multivariate gaussian techniques.
We also visualized the correlation of all the sensors (below) with each other to better understand any underlying connection between them. The numbers highlighted in shades of green, blue, and purple indicate that the sensors they represent are highly correlated. This means that they have a strong relationship with each other. If a sensor detects an anomaly, it is likely that the remaining sensors with which it is strongly correlated will also detect the anomaly.
 


