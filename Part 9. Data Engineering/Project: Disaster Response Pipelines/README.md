# Disaster Response Pipeline Project
> Goal: Building a ML pipeline to categorize emergency messages based on the needs communicated by the sender <br>
> Required skills: data pipelines, NLP pipelines, machine learning pipelines, supervised learning

### Table of Contents

### 1. Project Overview
In this project, data engineering, natural language processing, and machine learning skills are used to analyze message data that people sent during disasters to build a model for an API that classifies disaster messages. These messages could potentially be sent to appropriate disaster relief agencies.

### 2. Software Requirements (Installation)
  * Python 3.6 or above
  * nltk package: punkt, wordnet, and stopwords
  * pandas

### 3. File Descriptions
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
  |- InsertDatabaseName.db   # database to save clean data to
  
  - models
  |- train_classifier.py
  |- classifier.pkl  # saved model 
  
  - README.md

  ```

### 4. Project Components
  1. ETL Pipeline  
     In a Python script, `process_data.py`, write a data cleaning pipeline that:  
     * Loads the `messages` and `categories` datasets  
     * Merges the two datasets
     * Cleans the data
     * Stores it in a SQLite database

  2. ML Pipeline
     In a Python script, `train_classifier.py`, write a machine learning pipeline that:
      * Loads data from the SQLite database
      * Splits the dataset into training and test sets
      * Builds a text processing and machine learning pipeline
      * Trains and tunes a model using GridSearchCV
      * Outputs results on the test set
      * Exports the final model as a pickle file

  3. Flask Web App
     In a Python script, `run.py`, you'll need to:
      * Modify file paths for database and model as needed
      * Add data visualizations using Plotly in the web app

     cf. How to create and access the web app
     ```
     # Terminal
     python run.py
     env | grep WORK
     
     # In a browser
     https://SPACEID-3001.SPACEDOMAIN

     ```

### 5. Conclusion

### 6. Credits and Acknowledgements
  1. [Udacity]()
  2. [FigureEight](https://www.figure-eight.com/)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app. `python run.py`

3. Go to http://0.0.0.0:3001/
