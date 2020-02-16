# Capstone Project: Create a Customer Segmentation Report for Arvato Financial Services
> Goal: Use unsupervised learning techniques to perform customer segmentation. Then use supervised learning techniques to develop models which predict which individuals are most likely to convert into becoming customers for the company. <br>


## Table of Contents
- [Project Definition](#definition)
  - [Project Overview](#overview)
  - [Metrics](#metrics)
- [Software Requirements](#software)
- [File Descriptions](#hierarchy)
- [Project Components](#components)
  - [Data Analysis and Preprocessing](#data_analysis)
  - [Unsupervised Learning Models](#unsupervised_learning)
  - [Supervised Learning Models](#supervised_learning)
  - [Kaggle Competition (Results)](#kaggle_competition)
- [Conclusion](#conclusion)
  - [Reflection](#reflection)
  - [Improvement](#improvement)
- [Credits and Acknowledgements](#credits)

<a id='definition'></a>

## 1. Project Definition

<a id='overview'></a>

### 1.1. Project Overview
In this project, unsupervised and supervised learning methods are used to analyze customers' data to build a model that predicts potential customers of a mail-order sales company in Germany. The data was provided by Arvato Financial Solutions, a Bertelsman subsidiary company. 

First, customer segmentation was done by applying unsupervised learning techniques, specifically `Principal Component Analysis (PCA)` and `K-Means Clustering` methods. A comparison between the customers and the general population was made to uncover the demographic characteristics of the core customers.

Then, using another dataset from a marketing campaign of the company and applying supervised learning techniques, specifically classification algorithms, I built models to predict individuals that are most likely to be the potential customer of the company. To evaluate the models, `Area Under Curve (AUC)` for the ROC curve was used. 

Finally, I participated in a Kaggle competition to evaluate the supervised learning model.

<a id='metrics'></a>

### 1.2. Metrics

A [Receiver Operating Characteristic (ROC) curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) is created by plotting the true positive rate (TPR, the proportion of actual customers that are labeled as so) against the false positive rate (FPR, the proportion of non-customers labeled as customers). The plot shows the performance of a classification model at all classification thresholds. Area Under Curve (AUC) provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example. Using AUC as an evaluation metric is desirable for the following two reasons:
  * Scale-invariant: AUC measures how well predictions are ranked, rather than their absolute values.
  * Classification-threshold-invariant: AUC measures the quality of the model's predictions irrespective of what classification threshold is chosen.


<a id='software'></a>

## 2. Software Requirements (Installation)
  For detailed information, check `requirement.txt`
  * Anaconda distribution of Python (Python 3.6x or above)
  * scikit-learn
  * xgboost
  * lightgbm
  
  
<a id='hierarchy'></a>

## 3. File Descriptions
  Due to the sensitive nature of dataset, it is not made publicly available to the general public.
  ```
  - models_and_results # pickled models and prediction results

  - Arvato Project Workbook.html
  - Arvato Project Workbook.ipynb
  - README.md
  - requirement.txt
  ```


<a id='components'></a>

## 4. Project Components

<a id='data_analysis'></a>

###  4.1. Data Analysis and Preprocessing
This part refers to `Part 0: Get to Know the Data` in the jupyter notebook. The following steps were taken:
  * Load the `azdias`, demographics data for the general population of Germany, and `customers` dataset, demographics data for customers of a mail-order company.
  * Using attributes information provided, deal with missing or unknown values
  * Drop columns based on the three conditions:
    * Features not described in the attributes information
    * Check the proportion of missing values by columns and drop features (columns) with more than 30% missing values
    * Check the correlation among features and drop columns with 90% or higher correlations
  * Check the proportion of missing values by rows and drop rows with more than 20% missing values
  * Drop extra columns to match the features between `azdias` and `customers` datasets.
  * Re-encode mixed and categorical features
    * Find numerical, categorical, ordinal, and mixed type columns using the description datasets provided: To find numerical types, I used `numerical` as a keyword, `type, typology, typification, classification, CAMEO, flag` for categorical types, and assumed others as ordinal and mixed types.
    * Split information in the mixed type columns: After exploring the result, I was able to find `PRAEGENDE_JUGENDJAHRE` holds two different information: decades and movement. Each information was extracted and separately stored in the dataset.
    * Transform to categorical variables: Instead of creating dummy variables, I decided transform object type columns only. At this stage, I have only two object type columns `CAMEO_DEU_2015` and `OST_WEST_KZ`. 
  * Filling missing values and scaling
    * Fill in missing values with `mode` value: To apply PCA, missing values needs to be filled. As the majority of variable type is either categorical or ordinal, I decided to use `mode` instead of using other descriptive values such as `median` or `mean`.
    * Scale each column using standard scaler: Scaling is needed to prevent few variables dominating in the model.

<a id='unsupervised_learning'></a>

###  4.2. Unsupervised Learning Models
This part refers to `Part 1: Customer Segmentation Report` in the jupyter notebook. The following steps were taken:
  * Use `Principal Component Analysis (PCA)` to reduce the dimension and select the number of features needed to explain about 80% of the total variance
  * Apply `K-Means clustering` to the PCA-transformed dataset and find the demographic characteristics of the core customers

<a id='supervised_learning'></a>

###  4.3. Supervised Learning Models
This part refers to `Part 2: Supervised Learning Model` in the jupyter notebook. <br>The goal is to predict which individuals are most likely to respond to a mailout campaign. The following steps were taken:
  * Develop prediction models using the following classification methods
    * Logistic Regression 
    * Random Forest 
    * AdaBoost
    * Gradient Boosting
    * XGBoost
    * LightBGM 
  * For each model, use `GridSearchCV` and `Bayesian Optimization` to train and optimize the model by using a five-fold validation method
  * Use `Area Under Curve (AUC)` for the ROC curve to evaluate the models


<a id='kaggle_competition'></a>

###  4.4 Kaggle Competition (Results)
This part refers to `Part 3: Kaggle Competition` in the jupyter notebook.  <br>After training and testing the performance of several models based on cross-validation results, I decided to use the `Gradient Boosting Classifier` model, which showed the best performance in the previous section, to predict which individuals are most likely to respond to a mailout campaign. 

With the training set, the AUC score of this model was 0.6285. And my Kaggle score was 0.65388. Given that the data was highly imbalanced, I believe the model's performance is fine.

![kaggle_result](kaggle_result.png)  

In addition to the Gradient Boosting model, the XGBoost model also showed similar performance (AUC = 0.65095). These results indicate that models based on gradient boosting algorithms are a very good fit for this problem, given that we are dealing with a very large dataset for a classification problem.


<a id='conclusion'></a>

## 5. Conclusion 

<a id='reflection'></a>

### 5.1. Reflection
This was my first time working on a real-life dataset, and it has been a great learning experience on how to approach the problem in a methodical approach. The most challenging part for me was mainly on getting the data cleansed and processed without losing the key information.

In the Customer Segmentation part, I have performed data pre-processing and used the PCA method combined with the K-Means Clustering algorithm to get the clusters within the general population and customers. The clusters obtained were compared, and I was able to observe the differences in the clusters allocation. By digging into these differences, I was able to figure out which segment of the general population would be the biggest customer segment for mail-order company and vice versa. With the understanding of this difference, a company could focus more on the biggest customer segment within the general population and then perform target mail-out campaign. This would increase the customer conversion rate while reducing marketing costs.

In the supervised learning part, I was able to predict the class probabilities of each individual in the testing set to become a customer. To deal with the large and imbalanced dataset, I tried several algorithms and found out that gradient boosting algorithms outperforms other algorithms. In the Kaggle competition, I achieved an AUC score of 0.65388.

<a id='improvement'></a>

### 5.2. Improvement
For simplicity, I kept the categorical- and ordinal-type variables as given in the original dataset. The underlying assumption is that these variables are representing the actual scale of difference. However, this may not be the case, and properly transforming these variables would help to improve the model. 

Additionally, feature engineering (e.g., using Featuretools, an open-source Python library for automated feature engineering) and fine-tuning the model would also help to improve the prediction performance.

----------------------------

The detailed explanation and results can be found at the post available [here](https://medium.com/@hyeukjung213/kaggle-competition-identification-of-customer-segments-and-finding-potential-customers-of-93b73271bdc0).


<a id='credits'></a>

## 6. Credits and Acknowledgements
  1. [Udacity](https://www.udacity.com/)
  2. Proejct data: [Arvato Financial Solutions](https://finance.arvato.com/en-us//)
  3. [Kaggle Competition](https://www.kaggle.com/c/udacity-arvato-identify-customers/data)

