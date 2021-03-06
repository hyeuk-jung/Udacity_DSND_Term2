# Disaster Response Pipeline Project
> Goal: Building a ML pipeline to categorize emergency messages based on the needs communicated by the sender <br>
> Required skills: data pipelines, NLP pipelines, machine learning pipelines, supervised learning

## Table of Contents
- [Project Overview](#overview)
- [Software Requirements](#software)
- [File Descriptions](#hierarchy)
- [Project Components](#components)
  - [ETL Pipeline](#etl)
  - [ML Pipeline](#ml)
  - [Flask Web App](#flask)
- [Instructions](#instructions)
- [Conclusion](#conclusion)
- [Credits and Acknowledgements](#credits)

<a id='overview'></a>

## 1. Project Overview
In this project, data engineering, natural language processing, and machine learning skills are used to analyze message data that people sent during disasters to build a model for an API that classifies disaster messages. These messages could potentially be sent to appropriate disaster relief agencies.


<a id='software'></a>

## 2. Software Requirements (Installation)
  * Anaconda distribution of Pytho (Python 3.6x or above)
    * For detailed information, check `requirement.txt`
  * From `nltk` package: `punkt`, `wordnet`, and `stopwords`
  

<a id='hierarchy'></a>

## 3. File Descriptions
  ```
  - app
  | - template
  | |- master.html  # main page of web app
  | |- go.html  # classification result page of web app
  |- run.py  # Flask file that runs app
  
  - data
  |- disaster_categories.csv  # data to process 
  |- disaster_messages.csv  # data to process
  |- process_data.py
  |- DisasterResponse.db   # database to save clean data
  
  - models
  |- train_classifier.py
  |- classifier.pkl  # saved model 
  
  - README.md
  ```


<a id='components'></a>

## 4. Project Components

<a id='etl'></a>

###  4.1. ETL Pipeline  
In a Python script, `process_data.py`, write a data cleaning pipeline that:  
  * Loads the `messages` and `categories` datasets  
  * Merges the two datasets
  * Cleans the data
  * Stores it in a SQLite database

<a id='ml'></a>

###  4.2. ML Pipeline
In a Python script, `train_classifier.py`, write a machine learning pipeline that:
  * Loads data from the SQLite database
  * Splits the dataset into training and test sets
  * Builds a text processing and machine learning pipeline
  * Trains and tunes a model using GridSearchCV
  * Outputs results on the test set
  * Exports the final model as a pickle file

<a id='flask'></a>

###  4.3. Flask Web App
In a Python script, `run.py`, you'll need to:
  * Modify file paths for database and model as needed
  * Add data visualizations using Plotly in the web app

  cf. How to create and access the web app
  ```
  # Terminal
  python run.py
  env | grep WORK
     
  # In a browser (accessing to the Udacity workspace)
  https://SPACEID-3001.SPACEDOMAIN

  ```


<a id='instructions'></a>

## 5. Instructions
  1. Run the following commands in the project's root directory to set up your database and model.
      - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        ![process_data](process_data.png)  
      - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        ![train_classifier_1](train_classifier_1.png)  
        ![train_classifier_2](train_classifier_2.png)  
        cf. It took about 10 minutes to train the classifier with grid search.

  2. Run the following command in the app's directory to run your web app. `python run.py`
     ![run_web_app](run_web_app.png)  

  3. Go to http://0.0.0.0:3001/ or https://SPACEID-3001.SPACEDOMAIN/
      - Initial page
        ![web_app_first](web_app_first.png)  
      - Classifying result 
        ![classifying_result](classifying_result.png)  


<a id='conclusion'></a>

## 6. Conclusion
Due to the data imbalance, `F1-micro` score was used as a scoring method in hyperparameter tuning. Only focusing on the accuracy of the model would be inappropriate as majority of categories have less than 10% of the sample. 


<a id='credits'></a>

## 7. Credits and Acknowledgements
  1. Udacity logo and starter codes: [Udacity](https://www.udacity.com/)
  2. Proejct data: [FigureEight](https://www.figure-eight.com/)


