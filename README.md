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

#### 1.	Data Cleaning and pre-processing
The dataset we used had a few abnormalities, like empty columns, repetitive entries, missing values, and more. So, we had to clean the dataset. We have also utilized an interesting concept/technique called pickling for this project. Pickling is basically a technique of converting a python object into a byte stream to be stored in the file and maintain program state across all sessions in the project. Our cleaned dataset has a shape of (219521, 52).

![image](https://user-images.githubusercontent.com/35283246/163791152-2733067d-6819-4e0a-a826-c0ab5ab31f5e.png)

 
#### 2.	Exploratory Data Analysis
Below we visualize the distribution of some of the sensors.
We can see most of the sensor readings follow a normal distribution which is intuitive and this observation lets us use multivariate gaussian techniques.
We also visualized the correlation of all the sensors (below) with each other to better understand any underlying connection between them. The numbers highlighted in shades of green, blue, and purple indicate that the sensors they represent are highly correlated. This means that they have a strong relationship with each other. If a sensor detects an anomaly, it is likely that the remaining sensors with which it is strongly correlated will also detect the anomaly.

![image](https://user-images.githubusercontent.com/35283246/163791166-7768f76f-7bea-4b79-acc9-7795d6e7099d.png)

Sensors 2-12 seem to have high positive correlation with values ranging from around 0.2 to 0.9 whereas sensor 37 has high negative correlation with sensors 13, 35, 36 respectively with correlation values around -0.5. 
As we aim to find anomalies in the machinery, our focus will be on the detection of the machines which have a predicted status of "Broken" and "Recovering". Below is an excerpt from the plots of the readings of 2 of the sensors with respect to time. The 'Broken' readings are marked with a red cross and the ' Recovering ' readings are highlighted in yellow to indicate the anomaly.

![image](https://user-images.githubusercontent.com/35283246/163791221-57dd6f07-adcf-46c8-8518-3add7c7f38b8.png)

As seen clearly from the above plots, the red marks, which represent the broken state of the pump, perfectly overlaps with the observed disturbances of the sensor reading. Now we have a pretty good intuition about how each of the sensor reading behaves when the pump is broken vs operating normally.

#### 3.	Feature Engineering
Feature Engineering is a machine learning technique that takes advantage of data to generate new variables that are originally not a part of the training dataset. Feature Engineering can be used to produce new features for supervised learning as well as unsupervised learning. The primary purpose of feature engineering is to simplify as well as speed up data transformations while improving model accuracy. In this step, we will scale the data and apply Principal Component Analysis (PCA) to extract the most important features to be further used in training models. It is computationally quite expensive to process the data of this size, (219521, 53), hence the reason for reducing the dimensionality with PCA.

![image](https://user-images.githubusercontent.com/35283246/163791274-d07051d9-478f-44ce-8354-286f8580f9a3.png)

It appears that the first two principal components are the most important as per the features extracted by the PCA in above importance plot. So as the next step, I will perform PCA with 2 components which will be my features to be used in the training of the base model and the unsupervised model.
Next, we checked the stationarity and autocorrelation of these two principal components just to be sure they are stationary and not autocorrelated. 

*_Stationarity Test : For this we used the Augmented Dickey Fuller Test. Running the Dickey Fuller test on the 1st principal component, we got a p-value of 5.4536849418486247e-05 which is very small number (much smaller than 0.05). Thus, we rejected the Null Hypothesis and say the data is stationary. We performed the same on the 2nd component and got a similar result. So both principal components are stationary._*

*_Autocorrelation Test : For this test, we used the ACF plot  to visually verify that there is no autocorrelation for the two principal components. As evident from the below plot our two components have no autocorrelation._* 

![image](https://user-images.githubusercontent.com/35283246/163791496-0d7375e1-4237-4412-81ef-976a12a45018.png) ![image](https://user-images.githubusercontent.com/35283246/163791511-2c6d71e0-88f4-4fd4-954e-6e8b42ddc862.png)

#### 4.	Statistical Modelling
Statistical modelling can be defined as the process of applying statistical analysis to a dataset. It uses mathematical models and statistical assumptions to generate sample data and make predictions. We have used the following statistical modelling techniques –

*_4.1 BASE Model : Inter Quartile Range_* <br>

The interquartile range is the difference between the third quartile and the first quartile of the distribution, and it is used to measure variability. It helps us get an estimate of how wide our distribution is.

Now, let us walk through how we used Inter Quartile range to our advantage in this project. First, we calculated the inter quartile range which is the difference between Q3 and Q1. We then calculated the upper bound and the lower bound i.e. 1.5 times of IQR to mark the outliers. As we have now created the upper and lower bounds, we are able to classify any data points out of the bounds as outliers/anomalies. So, any data points that fell outside the upper and lower bounds were flagged and marked as anomalies/outliers. 

Next, we implemented Univariate feature selection to select k most important features. We used the Chi-Square test to identify the 3 most important sensors in our dataset. These are as below:

                Feature         Score
11  sensor_11  10106.761967<br>
12  sensor_12   9879.052739<br>
4   sensor_04   8167.176442<br>

Lastly, we plotted the abnormalities on the time series data as seen in the figure below for sensor 11.

![image](https://user-images.githubusercontent.com/35283246/163791733-93fcc32f-960b-4370-a4fa-faccadb3ec25.png)

As seen from plots, there are a lot more outliers in pc1 (1st principal component) than that from pc2. The outliers in pc1 represent approximately 14% of the data set. Also the outliers in pc1 seem to better explain the failures in the sensor readings from one of the sensors, sensor_00 is used in this case.

*_4.2 K-means Clustering_*<br>

The underline assumption in the clustering-based anomaly detection is that if we cluster the data, normal data will belong to clusters while anomalies will not belong to any clusters or belong to small clusters. We used the following steps to find and visualize anomalies.

*Calculate the distance between each point and its nearest centroid. The biggest distances are considered as anomaly.*<br>
*We use outliers_fraction to provide information to the algorithm about the proportion of the outliers present in our data set. Situations may vary from data set to data set. However, as a starting figure, I estimate outliers_fraction=0.14 (14% of dataset are outliers from base model)*<br>
*Calculate number_of_outliers using outliers_fraction.*<br>
*Set threshold as the minimum distance of these outliers.*<br>
*The anomaly result of anomaly1 contains the above method Cluster (0:normal, 1:anomaly).*<br>
*Visualize anomalies with cluster view.*<br>
*Visualize anomalies with Time Series view.*<br>

![image](https://user-images.githubusercontent.com/35283246/163792170-96eacc08-cf5b-4115-a377-edc598d2074c.png)

*4.3 Multivariate Gaussian Distribution*<br>

Multivariate Gaussian Distributions (multi-variate) are high dimensional normal distributions (univariate). A vector is said to be multi-variate normally distributed if all the linear combinations of its components follow normal distributions. We implemented this model on 5 of the principal components. The model learns by estimating the parameters of the distribution and assigning observations to the distribution based on the probabilities. Below is the distribution of our first 5 principal components:

![image](https://user-images.githubusercontent.com/35283246/163792248-664a796f-871e-488b-88e4-a45db7d5d2ed.png)

The 5 principal components chosen have near to normal distributions, all centred around 0. Each distribution is defined by its own set of parameters-mean and covariance matrices.

After we calculated the mean and co-variance matrix of the five principal components, we applied a multivariate gaussian distribution on the components collectively. We defined a threshold called “epsilon” which we use to determine if an observation should be flagged as an anomaly or not. After we have set the multivariate distribution, we implemented a search algorithm using the F1 score to pick the best threshold for flagging an observation as an anomaly. 

The best value for “epsilon” is determined using a stepwise iterative process. The size of the step is determined from the maximum and minimum values of the probabilities of the observations (which are determined by the multivariate normal random variable). This step is used to iterate through the range of probabilities generated, each of which is a potential epsilon. Predictions are made at every stage on the test data for every potential value of the epsilon for which the F-score is calculated. The epsilon with the highest F-score is chosen as the “best” epsilon. 

Next, we will flag an observation as an anomaly if the probability of that observation to be a part of the dataset is less than the determined threshold. We got the best threshold value as  1.7111345161379106e-17.

*4.4 Gaussian Mixture Model*<br>

Gaussian Mixture Model can be considered a “generalized” version of k-means algorithm where clustering is done using probability measures. The Gaussian Mixture Model used applies an iterative EM (Expectation Maximization) algorithm to fit a mixture of Gaussian models where the following steps are repeated until convergence occurs:

•	E-Step- “Soft clustering” is performed. This means that there are no restrictions on the number of clusters a datapoint may belong to (unlike k-means which performs hard clustering and assigns a datapoint to only one cluster) and considers the possibility that clusters may overlap (mixed membership). This step returns the probability of each point belonging in a certain cluster (in our case, anomaly or not)<br>

•	M-Step-  updates the membership and parameters of the clusters

We used the model for anomaly detection in 2 ways:
1) Predicting the labels from the model generated<br>
2) Using the predicted probabilities from the model to detect anomalies. This was implemented using the same threshold mechanism as used in the Multivariate Gaussian Distribution model.<br>

#### 5.	Evaluation and Comparison of Models 
For this project as seen above, we have used 4 different models namely Interquartile Range, K-Means clustering, Multivariate Gaussian Distribution and Gaussian Mixture Model. Now, let us compare the results that we received from these three models. We have indexed 0: as a normal and well-functioning machine and 1: as an anomaly. 

![image](https://user-images.githubusercontent.com/35283246/163792590-ae2990d9-1d77-4309-9e3a-4b6322a84aef.png)

![image](https://user-images.githubusercontent.com/35283246/163792599-64361c87-5106-4b65-ac5b-0f618853df0f.png)

 we choose the Multivariate Gaussian Distribution model as our best model.

## Conclusion

So far, we have done anomaly detection with four different methods. In doing so, we went through most of the steps of the commonly applied Data Science Process which includes the following steps:

1.	Problem Identification
2.	Data Wrangling
3.	Exploratory Data Analysis
4.	Pre-processing and training data development
5.	Modeling
6.	Documentation

One of the challenges we faced during this project is that training anomaly detection models with unsupervised learning algorithms with such a large data set can be computationally very expensive. This limited us from implementing SVM (Support Vector Machine modelling) on this data as it was taking a very long time to train the model with no success. We suggest the following next steps on improving the model:

1.	Feature selection using advanced techniques
2.	Advanced hyperparameter tuning
3.	Implementing other learning algorithms such as SVM, DBSCAN, etc.

























 


