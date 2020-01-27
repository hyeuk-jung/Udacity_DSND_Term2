# Part 9. Data Engineering

## Part 9-1: ETL Pipelines
> ETL stands for extract, transform, and load. This is the most common type of data pipeline.

 ### 1. Data pipelines: ETL vs. ELT
    1) [ETL (Extract, Transform, Load)](https://en.wikipedia.org/wiki/Extract,_transform,_load)
        (1) 

    2) ELT (Extract, Load, Transform)

 ### 2. ETL pipeline
   1. Extract data from different sources such as: 
        * .csv files
        * .json files 
        * APIs
   2. Transform data
        * combining data from different sources
        * data cleaning
        * data types
        * parsing dates
        * file encodings
        * missing data: Drop rows/columns or impute values (fill with mean, median, forward, backward values)
        * duplicate data
        * dummy variables
        * remove outliers
        * scaling features (normalizing)
        * engineering features (eg. polynomial variables)
   3. Load data into database
        * SQL, csv, json
        * Other data storage systems: [Redis](https://redis.io/), [Cassandra](http://cassandra.apache.org/), [Hbase](http://hbase.apache.org/), [MongoDB](https://www.mongodb.com/)
        * [ranking of database engines](https://db-engines.com/en/ranking)
   4. Create an ETL pipeline


## Part 9-2: NPL Pipelines
> Natural language processing 

 ### 1. Text Processing

 ### 2. Modeling


## Part 9-3: Machine Learning Pipelines
> Using Scikit-learn package to code a ML pipeline.

 ### 1. Scikit-learn pipelines

 ### 2. Feature Union

 ### 3. Grid Search


## Project: Disaster Response Pipeline
> Goal: Building a ML pipeline to categorize emergency messages based on the needs communicated by the sender <br>
> Required skills: data pipelines, NLP pipelines, machine learning pipelines, supervised learning

