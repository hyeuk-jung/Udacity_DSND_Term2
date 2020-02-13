#!/usr/bin/env python
# coding: utf-8

# # Capstone Project: Create a Customer Segmentation Report for Arvato Financial Services
# 
# In this project, you will analyze demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population. You'll use unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. Then, you'll apply what you've learned on a third dataset with demographics information for targets of a marketing campaign for the company, and use a model to predict which individuals are most likely to convert into becoming customers for the company. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# If you completed the first term of this program, you will be familiar with the first part of this project, from the unsupervised learning project. The versions of those two datasets used in this project will include many more features and has not been pre-cleaned. You are also free to choose whatever approach you'd like to analyzing the data rather than follow pre-determined steps. In your work on this project, make sure that you carefully document your steps and decisions, since your main deliverable for this project will be a blog post reporting your findings.

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from datetime import datetime

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# Printing options (optional)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 60)
pd.set_option('display.max_colwidth', -1)
plt.rc("font", size = 14)
sns.set(style = "white")
sns.set(style = "whitegrid", color_codes = True)


# ## Part 0: Get to Know the Data
# 
# There are four data files associated with this project:
# 
# - `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
# - `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
# - `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
# - `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. **Use the information from the first two files to figure out how customers ("CUSTOMERS") are similar to or differ from the general population at large ("AZDIAS"), then use your analysis to make predictions on the other two files ("MAILOUT"), predicting which recipients are most likely to become a customer for the mail-order company**.
# 
# The "CUSTOMERS" file contains three extra columns ('CUSTOMER_GROUP', 'ONLINE_PURCHASE', and 'PRODUCT_GROUP'), which provide broad information about the customers depicted in the file. The original "MAILOUT" file included one additional column, "RESPONSE", which indicated whether or not each recipient became a customer of the company. For the "TRAIN" subset, this column has been retained, but in the "TEST" subset it has been removed; it is against that withheld column that your final predictions will be assessed in the Kaggle competition.
# 
# Otherwise, all of the remaining columns are the same between the three data files. For more information about the columns depicted in the files, you can refer to two Excel spreadsheets provided in the workspace. **[One of them](./DIAS Information Levels - Attributes 2017.xlsx) is a top-level list of attributes and descriptions, organized by informational category**. **[The other](./DIAS Attributes - Values 2017.xlsx) is a detailed mapping of data values for each feature in alphabetical order**.
# 
# In the below cell, we've provided some initial code to load in the first two datasets. Note for all of the `.csv` data files in this project that they're semicolon (`;`) delimited, so an additional argument in the [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call has been included to read in the data properly. Also, considering the size of the datasets, it may take some time for them to load completely.
# 
# You'll notice when the data is loaded in that a warning message will immediately pop up. Before you really start digging into the modeling and analysis, you're going to need to perform some cleaning. Take some time to browse the structure of the data and look over the informational spreadsheets to understand the data values. Make some decisions on which features to keep, which features to drop, and if any revisions need to be made on data formats. It'll be a good idea to create a function with pre-processing steps, since you'll need to clean all of the datasets before you work with them.

# # Preprocessing
# 
# ## 1. Drop columns
# ###   1) Non-overlapping columns
# ###   2) Columns with missing values ( > 20%)
# ###   3) Correlation ( > 90%)

# In[2]:


# load in the data
path = '../../data/Term2/capstone/arvato_data/'
azdias = pd.read_csv(path+'Udacity_AZDIAS_052018.csv', sep=';', low_memory=False)
customers = pd.read_csv(path+'Udacity_CUSTOMERS_052018.csv', sep=';', low_memory=False)


# In[3]:


dias_attributes = pd.read_excel("DIAS Attributes - Values 2017.xlsx")
dias_info = pd.read_excel("DIAS Information Levels - Attributes 2017.xlsx")
del dias_attributes['Unnamed: 0']
del dias_info['Unnamed: 0']

dias_attributes['Attribute'].fillna(method ='ffill', inplace = True) 
dias_attributes['Description'].fillna(method ='ffill', inplace = True) 


# In[4]:


# Be sure to add in a lot more cells (both markdown and code) to document your approach and findings!
# General
display(azdias.head())
display(azdias.info())

# Customers
display(customers.head())
display(customers.info())


# In[5]:


display(dias_attributes.head(10))
display(dias_attributes.info())

display(dias_info.head(10))
display(dias_info.info())


# In[6]:


def plot_null_ratio(df):
    ''' plot the top 50 columns' null value ratio
    args:
    df; (dataframe) the dataframe you want to check the ratio
    
    return:
    missing_ratio; (dataframe) descending-ordered values
    '''
    missing_ratio = df.isnull().sum() / df.shape[0]
    missing_ratio.sort_values(ascending = False, inplace = True)
    
    plt.figure(figsize = (16, 8))
    missing_ratio[:50].plot.bar()
    plt.title('Distribution of the Ratio of Missing Values in Each Column')
    plt.xlabel('Percentage of missing values')
    plt.ylabel('Count')
    plt.show()
    
    return missing_ratio


# In[7]:


def data_cleaning(azdias, customers):
    ''' drop columns with missing values and highly-correlated columns
    args:
    
    return:
    
    '''
    # 1. Drop non-overlapping columns
    print('1. Drop non-overlapping columns')
    azdias_unique_col = list(set(azdias.columns) - set(customers.columns))
    customers_unique_col = list(set(customers.columns) - set(azdias.columns))
    azdias.drop(azdias_unique_col, axis = 1, inplace = True)
    customers.drop(customers_unique_col, axis = 1, inplace = True)
    print('From azdias: ', azdias_unique_col, '\nFrom customers: ', customers_unique_col)
    display(str(datetime.now()))

    # 2. Drop columns with missing values > 20% in both data sets
    ## Check missing value ratio of each column and plot top 50
    print('2. Drop columns with missing values greater than 20% in both datasets')
    azdias_null = plot_null_ratio(azdias)
    customers_null = plot_null_ratio(customers)
    null_cols = list(azdias_null.index[azdias_null > 0.2] & customers_null.index[customers_null > 0.2])
    ## Drop columns
    azdias.drop(null_cols, axis = 1, inplace = True)
    customers.drop(null_cols, axis = 1, inplace = True)
    print('Columns dropped from both datasets: ', null_cols)
    display(str(datetime.now()))
    
    # 3. Drop highly correlated columns
    print('3. Drop highly correlated columns (corr > 90%)')
    azdias_corr = azdias.corr()
    customers_corr = customers.corr()
    plt.figure(figsize = (10, 10)); sns.heatmap(azdias_corr, cmap = "YlGnBu"); plt.show();
    plt.figure(figsize = (10, 10)); sns.heatmap(customers_corr); plt.show();
    ## Select upper triangle of correlation matrix
    azdias_corr_upper = azdias_corr.where(np.triu(np.ones(azdias_corr.shape), k = 1).astype(np.bool))
    customers_corr_upper = customers_corr.where(np.triu(np.ones(customers_corr.shape), k = 1).astype(np.bool))
    ## Find index of feature columns with correlation greater than 0.9
    azdias_to_drop = [column for column in azdias_corr_upper.columns if any(azdias_corr_upper[column] > 0.9)]
    customers_to_drop = [column for column in customers_corr_upper.columns if any(customers_corr_upper[column] > 0.9)]
    ## Drop columns
    correlated_col = list(set(azdias_to_drop + customers_to_drop))
    azdias.drop(correlated_col, axis = 1, inplace = True)
    customers.drop(correlated_col, axis = 1, inplace = True)
    display(str(datetime.now()))
    
    # 4. Change two object type columns to numeric type
    ## CAMEO_INTL_2015, EINGEFUEGT_AM info unavailable in the attribute data
    print('4. Deal with unknown values')
    azdias_object_col = azdias.select_dtypes(include = ['object']).columns.tolist() 
    print('[Before transforming]')
    for col in azdias_object_col:
        print(col, ':', azdias[col].unique())
        
    azdias.CAMEO_DEUG_2015.replace([np.nan, 'X'], -1, inplace = True)
    azdias.CAMEO_INTL_2015.replace([np.nan, 'XX'], -1, inplace = True)
    to_numeric_col = ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015']
    azdias[to_numeric_col] = azdias[to_numeric_col].apply(pd.to_numeric)
    print('[After transforming]')
    for col in azdias_object_col:
        print(col, ':', azdias[col].unique())
        
    # display(dias_attributes[dias_attributes.Attribute.isin(azdias_object_col)])
    # azdias.CAMEO_DEU_2015.replace([np.nan, 'XX'], -1, inplace = True)
    
    temp = dias_attributes[dias_attributes.Meaning == 'unknown']
    for col in azdias.columns:
        unknown_val = temp[temp.Attribute == col].Value.tolist()
        if len(unknown_val) != 0:
            azdias[col].replace(unknown_val + ['X', 'XX', np.nan], -1, inplace = True)
    display(str(datetime.now()))
        
#     for col in azdias.columns[:50]:
#         print(azdias[col].unique())

    return azdias, customers


# In[8]:


azdias, customers = data_cleaning(azdias, customers)


# In[10]:


azdias.replace([np.nan, 'X', 'XX'], -1, inplace = True)
azdias_missing_ratio = azdias.isnull().sum() / azdias.shape[0]
azdias_missing_ratio.sort_values(ascending = False, inplace = True)
# display(azdias_missing_ratio[:50])

plt.figure(figsize = (16, 8))
azdias_missing_ratio[:50].plot.bar()
plt.title('Distribution of the Ratio of Missing Values in Each Column')
plt.xlabel('Percentage of missing values')
plt.ylabel('Count')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


display(azdias.select_dtypes(include=['object']).columns)
display(customers.select_dtypes(include=['object']).columns)

display(list(set(azdias.columns) - set(customers.columns)))
display(list(set(customers.columns) - set(azdias.columns)))

# customers.drop(['ONLINE_PURCHASE', 'PRODUCT_GROUP', 'CUSTOMER_GROUP'], axis = 1, inplace = True)


# In[16]:


# Check missing value ratio of each column and plot top 50
azdias_null = azdias.isnull().sum() / azdias.shape[0]
azdias_null.sort_values(ascending = False, inplace = True)

plt.figure(figsize=(16, 8))
azdias_null[:50].plot.bar()
plt.title('Distribution of the Ratio of Missing Values in Each Column')
plt.xlabel('Percentage of missing values')
plt.ylabel('Count')
plt.show()


# In[6]:


# Check missing value ratio of each column and plot top 50
customers_null = customers.isnull().sum() / customers.shape[0]
customers_null.sort_values(ascending = False, inplace = True)

plt.figure(figsize=(16, 8))
customers_null[:50].plot.bar()
plt.title('Distribution of the Ratio of Missing Values in Each Column')
plt.xlabel('Percentage of missing values')
plt.ylabel('Count')
plt.show()


# In[13]:


display(list(missing_ratio.index[missing_ratio > 0.2] & 
             missing_ratio_customers.index[missing_ratio_customers > 0.2]))


# In[14]:


# # drop non-overlapping columns
# azdias_unique_col = list(set(azdias.columns) - set(customers.columns))
# customers_unique_col = list(set(customers.columns) - set(azdias.columns))
# azdias.drop(azdias_unique_col, axis = 1, inplace = True)
# customers.drop(customers_unique_col, axis = 1, inplace = True)
# print(list(set(azdias.columns) - set(customers.columns)), list(set(customers.columns) - set(azdias.columns)))

# # drop columns with missing values > 20% in both data sets
# print(azdias.shape[1], customers.shape[1])
# null_cols = list(missing_ratio.index[missing_ratio > 0.2] & 
#                  missing_ratio_customers.index[missing_ratio_customers > 0.2])
# azdias.drop(null_cols, axis = 1, inplace = True)
# customers.drop(null_cols, axis = 1, inplace = True)
# print(azdias.shape[1], customers.shape[1])


# In[23]:


plt.figure(figsize = (16, 16))
azdias_corr = azdias.corr()
sns.heatmap(azdias_corr, cmap = "YlGnBu")
plt.show()


# In[26]:


# Select upper triangle of correlation matrix
azdias_corr_upper = azdias_corr.where(np.triu(np.ones(azdias_corr.shape), k = 1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.9
azdias_to_drop = [column for column in azdias_corr_upper.columns if any(azdias_corr_upper[column] > 0.9)]
display(azdias_to_drop)


# In[27]:


plt.figure(figsize = (16, 16))
customers_corr = customers.corr()
sns.heatmap(customers_corr) # , cmap = "YlGnBu"
plt.show()

customers_corr_upper = customers_corr.where(np.triu(np.ones(customers_corr.shape), k = 1).astype(np.bool))
customers_to_drop = [column for column in customers_corr_upper.columns if any(customers_corr_upper[column] > 0.9)]
display(customers_to_drop)


# In[31]:


correlated_col = list(set(azdias_to_drop + customers_to_drop))
display(correlated_col)
azdias.drop(correlated_col, axis = 1, inplace = True)
customers.drop(correlated_col, axis = 1, inplace = True)


# In[32]:


display(azdias.select_dtypes(include=['object']).head())
display(customers.select_dtypes(include=['object']).head())


# In[33]:


dias_attributes = pd.read_excel("DIAS Attributes - Values 2017.xlsx")
dias_info = pd.read_excel("DIAS Information Levels - Attributes 2017.xlsx")
del dias_attributes['Unnamed: 0']
del dias_info['Unnamed: 0']


# In[34]:


display(dias_attributes.head(10))
display(dias_attributes.info())


# In[45]:


dias_attributes['Attribute'].fillna(method ='ffill', inplace = True) 
dias_attributes['Description'].fillna(method ='ffill', inplace = True) 

azdias_object_col = azdias.select_dtypes(include=['object']).columns.tolist()
display(dias_attributes[dias_attributes.Attribute.isin(azdias_object_col)])
# CAMEO_INTL_2015, EINGEFUEGT_AM info unavailable in the attribute data


# In[52]:


for col in azdias_object_col:
    print(col, ':', azdias[col].unique())


# In[53]:


azdias.CAMEO_DEU_2015.replace([np.nan, 'XX'], -1, inplace = True)
azdias.CAMEO_DEUG_2015.replace([np.nan, 'X'], -1, inplace = True)
azdias.CAMEO_INTL_2015.replace([np.nan, 'XX'], -1, inplace = True)


# In[55]:


to_numeric_col = ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015']
azdias[to_numeric_col] = azdias[to_numeric_col].apply(pd.to_numeric)
for col in azdias_object_col:
    print(col, ':', azdias[col].unique())


# In[67]:


temp = dias_attributes[dias_attributes.Meaning == 'unknown']

for col in azdias.columns:
    unknown_val = temp[temp.Attribute == col].Value.tolist()
    if len(unknown_val) != 0:
        azdias[col].replace(unknown_val + ['X', 'XX', np.nan], -1, inplace = True)


# In[68]:


for col in azdias.columns[:50]:
    print(azdias[col].unique())


# In[ ]:





# In[58]:


# display(dias_attributes[dias_attributes.Meaning == 'unknown'])


# In[35]:


display(dias_info.head(10))
display(dias_info.info())


# In[26]:


dias_info['Information level'].unique()


# In[27]:


dias_info[dias_info['Information level'].notnull()]


# ## Preprocessing
# 
# ### DIAS Attributes

# In[26]:


# dias_attributes.loc[:, ['Attribute', 'Description']].fillna(method = 'ffill', axis = 1, inplace = True)
dias_attributes['Attribute'].fillna(method ='ffill', inplace = True) 
dias_attributes['Description'].fillna(method ='ffill', inplace = True) 
display(dias_attributes.head(10))
display(dias_attributes.info())


# In[27]:


dias_attributes[dias_attributes.isnull().any(axis = 1)]


# In[8]:





# In[ ]:





# ## Part 1: Customer Segmentation Report
# 
# The main bulk of your analysis will come in this part of the project. Here, you should use unsupervised learning techniques to describe the relationship between the demographics of the company's existing customers and the general population of Germany. By the end of this part, you should be able to describe parts of the general population that are more likely to be part of the mail-order company's main customer base, and which parts of the general population are less so.

# In[ ]:





# ## Part 2: Supervised Learning Model
# 
# Now that you've found which parts of the population are more likely to be customers of the mail-order company, it's time to build a prediction model. Each of the rows in the "MAILOUT" data files represents an individual that was targeted for a mailout campaign. Ideally, we should be able to use the demographic information from each individual to decide whether or not it will be worth it to include that person in the campaign.
# 
# The "MAILOUT" data has been split into two approximately equal parts, each with almost 43 000 data rows. In this part, you can verify your model with the "TRAIN" partition, which includes a column, "RESPONSE", that states whether or not a person became a customer of the company following the campaign. In the next part, you'll need to create predictions on the "TEST" partition, where the "RESPONSE" column has been withheld.

# In[ ]:


mailout_train = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_MAILOUT_052018_TRAIN.csv', sep=';')


# In[ ]:





# ## Part 3: Kaggle Competition
# 
# Now that you've created a model to predict which individuals are most likely to respond to a mailout campaign, it's time to test that model in competition through Kaggle. If you click on the link [here](http://www.kaggle.com/t/21e6d45d4c574c7fa2d868f0e8c83140), you'll be taken to the competition page where, if you have a Kaggle account, you can enter. If you're one of the top performers, you may have the chance to be contacted by a hiring manager from Arvato or Bertelsmann for an interview!
# 
# Your entry to the competition should be a CSV file with two columns. The first column should be a copy of "LNR", which acts as an ID number for each individual in the "TEST" partition. The second column, "RESPONSE", should be some measure of how likely each individual became a customer – this might not be a straightforward probability. As you should have found in Part 2, there is a large output class imbalance, where most individuals did not respond to the mailout. Thus, predicting individual classes and using accuracy does not seem to be an appropriate performance evaluation method. Instead, the competition will be using AUC to evaluate performance. The exact values of the "RESPONSE" column do not matter as much: only that the higher values try to capture as many of the actual customers as possible, early in the ROC curve sweep.

# In[ ]:


mailout_test = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_MAILOUT_052018_TEST.csv', sep=';')


# In[ ]:




