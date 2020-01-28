# Part 9. Data Engineering

## Part 9-1: ETL Pipelines
> ETL stands for extract, transform, and load. This is the most common type of data pipeline.

 ### 1. Data pipelines: ETL vs. ELT
   1. [ETL (Extract, Transform, Load)](https://en.wikipedia.org/wiki/Extract,_transform,_load)

   2. ELT (Extract, Load, Transform)

 ### 2. ETL Pipeline
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
      * [Ranking of database engines](https://db-engines.com/en/ranking)

   4. Create an ETL pipeline


## Part 9-2: NPL Pipelines
> Natural language processing 

 ### 1. Text Processing
   > Goal: Take raw input text, clean it, normalize it, and convert it into a form that is suitable for feature extraction.  
      1) **Extracting plain text** that is free of any source specific markdup or constructs that are not relevant to the task.  
      2) **Reducing complexity** by dealing with capitalization, punctuation, and commno words such as _a, of, and the_.

   1. Cleaning: Removing irrelevant items, such as HTML tags
      * Package `bs4 (BeautifulSoup)`

   2. Normalization: Converting to all lowercase and removing punctuation (or replacing with a space)
      * Built-in functions
        - `string_name.lower()`: Replaces all characters to lowercase
        - `string_name.upper()`: Replaces all characters to uppercase
      * Package `Re`
        - `^`: Negation expression
        - `[a-zA-Z0-9]`: String that has a letter from a-z, A-Z, or 0-9
        - [Other examples](https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285)

   3. Tokenization: Splitting text into words or tokens and remove common words
      * Built-in functions
        - `string_name.split()`: By default, it splits a string based on spaces, tabs, and new lines
      * Package [`nltk.tokenize`](http://www.nltk.org/api/nltk.tokenize.html)
        - `word_tokenize(string_name)`: Splits text into words
        - `sent_tokenize(string_name)`: Splits text into sentences

   4. Stop Word Removal: Removing words that are too common or uninformative such as _a, an, the, in, at, on_
      * Package `nltk.corpus`
        - `stopwords.words('language_name')`: Returns a list of stop words in the given language

   5. Part of Speech Tagging
      * Part-of-speech tagging using a predefined grammar like this is a simple, but limited, solution. There are other more advanced forms of POS tagging that can learn sentence structures and tags from given data, including _Hidden Markov Models (HMMs)_ and _Recurrent Neural Networks (RNNs)_.
      * Package `nltk`
        - `pos_tag(word_tokenize(string_name))`: Returns a list of tuples containing a tag for each word identifying different parts of speech
      * Sentence parsing
        - `new_grammer = nltk.CFG.fromstring('customized_grammar')` -> `parser = nltk.ChartParser(new_grammer)`: Defining a custom grammar and using it as a parser
        - `for tree in parser.parse(word_tokenize(string_name)): tree.draw()`: Returns a tree of POS tagging based on the predefined grammar
      
   6. Named Entity Recognition
      * Package `nltk`
        - `ne_chunk(pos_tag(word_tokenize(string_name)))`: Recognizes named entities in a tagged sentence

   7. Stemming and Lemmatization: Converting words into their dictionary forms
      * Stemming: A process of reducing a word to its stem or root form. It often returns incomplete words
      * Package `nlkt.stem.porter`
        - `PorterStemmer().stem('word_name')`
        - `SnowballStemmer().stem('word_name')`
      * Lemmatization: A process which transforms words in different variations to their roots by mapping them using a dictionary
      * Package `nlkt.stem.wordnet`
        - `WordNetLemmatizer().lemmatize('word_name', pos = 'v')`: Based on the POS, it transforms corresponding words, in this case verbs to their roots (default: pos = 'n' --> transforms nouns)
    
 ### 2. Feature Extraction
   > Goal: Extract and produce feature representations that are appropriate for the type of NLP task you are trying to accomplish and the type of model you are planning to use.  
   [WordNet visualization tool](http://mateogianolio.com/wordnet-visualization/)
   
   1. Bag of Words: Using a bag of words (or list of words) from the text processing, create a vector (document-term matrix) which illustrates the relationship between documents in rows and words or terms in columns. Each cell in the matrix represents the term frequency.
      * Usage: Comparing term similaries via calculating dot product or cosine similarity of two sentences
      * Limitation: It treats every word as being equally important

   2. TF-IDF: In addition to the document-term matrix, this uses the document frequency, the sum of term frequencies of each word. By dividing each term frequency by the document frequency of the corresponding term, it gives us a metric that is proportional to the frequency of occurence of a term in a document. (These values are inversely proportional to the number of documents it appears in.)
      * TF-IDF(t, d, D) = tf(t, d) * idf(t, D) where t for term, d for document, and D for the total number of documents in the collection D
        - tf(t, d): Term frequency, ![equation](https://latex.codecogs.com/gif.latex?count%28t%2C%20d%29%20%5Cdiv%20%7Cd%7C) (a term t in a document d / total number of term in document d)
        -  idf(t, D): Inverse document frequency, ![equation](https://latex.codecogs.com/gif.latex?%5Clog%28%7CD%7C%20%5Cdiv%20%7C%7Bd%20%5Cin%20D%20%3A%20t%20%5Cin%20d%7D%7C%29) (log(D / the number of documents where term t is present))
      * Usage: Assigns weights to words that signify their relevance in documents

   3. Word Embeddings: Using a vector space, find an embedding for each word. It also conveys relationships among other words. This is useful when using one-hot encoding or other ways of numerical transformation is inefficient.
      * Usage: Finding synonyms and analogies, identifying concepts around which words are clustered, classifying words as positive, negative, and neutral, etc.
      * Word2Vec
        - Idea: A model is able to predict a given word given neighboring words (Continuous Bag of Words, CBoW), or vice versa (Skip-gram). Predicting neighboring words for a given word is likely to capture the contextual meaning of words well. 
        - Properties
          * Forward embedding
          * It yields a robust and distributed representation of words
          * Vector size is independent from vocabulary you train on
          * Train once (which returns pre-trained embedding) and store in a lookup table so it can be reused 
          * It is ready to be used in deep learning architectures
      * GloVe (Global Vectors for World Representation)
        - Idea: Directly optimize the vector representation of each word using co-occurrence statistics
      * t-SNE (t-Distributed Stochastic Neighbor Embedding), a dimensionality reduction technique
        - Idea: For visualizing word embeddings, dimensionality reduction is needed while preserving the linear substructures (similarities and differences or relationships learned by the model) 
        
 ### 3. Modeling
   > Goal: Design a statistical or machine learning model, fit its parameters to training data, use an optimization procedure, and then use it to make predictions about unseen data.


## Part 9-3: Machine Learning Pipelines
> Using Scikit-learn package to code a ML pipeline.  
  Advantages: 

 ### 1. Scikit-learn pipeline
   1. Without pipeline
      ```python
      vect = CountVectorizer()
      tfidf = TfidfTransformer()
      clf = RandomForestClassifier()

      # train classifier
      X_train_counts = vect.fit_transform(X_train)
      X_train_tfidf = tfidf.fit_transform(X_train_counts)
      clf.fit(X_train_tfidf, y_train)

      # predict on test data
      X_test_counts = vect.transform(X_test)
      X_test_tfidf = tfidf.transform(X_test_counts)
      y_pred = clf.predict(X_test_tfidf)
      ```
      * Estimator: An estimator is any object that learns from data, whether it's a classification, regression, or clustering algorithm, or a transformer that extracts or filters useful features from raw data. Since estimators learn from data, they each must have a `fit` method that takes a dataset. 
        * Example: `CountVectorizer`, `TfidfTransformer`, and `RandomForestClassifier`

      * Transformer: A transformer is a specific type of estimator that has a `fit` method to learn from _training data_, and then a `transform` method to apply a transformation model to new data (or a `fit_transform` method). These transformations can include cleaning, reducing, expanding, or generating features.
        * Example: `CountVectorizer` and `TfidfTransformer`

      * Predictor: A predictor is a specific type of estimator that has a `predict` method to predict on _test data_ based on a supervised learning algorithm, and has a `fit` method to train the model on training data. 
        * Example: `RandomForestClassifier`

   2. Using pipeline
      ```python
      pipeline = Pipeline([
          ('vect', CountVectorizer()),
          ('tfidf', TfidfTransformer()),
          ('clf', RandomForestClassifier())
      ])

      # train classifier
      pipeline.fit(Xtrain)

      # evaluate all steps on test set
      predicted = pipeline.predict(Xtest)
      ```
      * Creating a pipeline: Make a list of (key, value) pairs, where the key is a string containing what you want to name the step, and the value is the estimator object. It combines a list of estimators to become a single estimator and runs its estimators in a sequence.
        * `Pipeline([ ('key_1', CountVectorizer()), ('key_2', TfidfTransformer()) ])`

      * How to use a pipeline: Pipeline takes on all the methods of whatever the last estimator in its sequence is. 
        * Example: `pipeline.fit(Xtrain)` automatically calls fit and transform methods of each estimator or transformer in the pipeline and uses the result on the final estimator's method.

   3. Advantages of using pipeline
      * Simplicity and convenience
        * Automates repetitive steps and the intermediate actions required to execute each step
        * Easilay understandable workflow
        * Reduces mental workload

      * Optimizing entire workflow by running grid search on a pipeline
        * Grid search: A method that automates the process of testing different hyperparameters to optimize a model (hyperparameter tuning)

      * Preventing data leakage
        * Using pipeline, all transformations for data preparation and feature extractions occur within each fold of the cross validation process. This prevents common mistakes such as training dataset being affected by test data.


 ### 2. Feature Union
   1. Feature Union: A class in scikit-learn’s Pipeline module that allows us to perform steps in parallel and take the union of their results for the next step.
     * A pipeline performs a list of steps in a _linear sequence_, while a feature union performs a list of steps in _parallel_ and then combines their results. In more complex workflows, multiple feature unions are often used within pipelines, and multiple pipelines are used within feature unions.

   2. Using feature union
      ```python
      X = df['text'].values
      y = df['label'].values
      X_train, X_test, y_train, y_test = train_test_split(X, y)

      pipeline = Pipeline([
          ('features', FeatureUnion([

              ('nlp_pipeline', Pipeline([
                  ('vect', CountVectorizer()
                  ('tfidf', TfidfTransformer())
              ])),

              ('txt_len', TextLengthExtractor())
          ])),

          ('clf', RandomForestClassifier())
      ])

      # train classifier
      pipeline.fit(Xtrain) # Runs 'features': nlp_pipeline and txt_len estimators run in parallel

      # predict on test data
      predicted = pipeline.predict(Xtest)
      ```
      * Creating a feature union: Make a list of (key, value) pairs, where the key is a string containing what you want to name the step, and the value is the estimator object. It combines a list of estimators to become a single estimator and runs its estimators in parallel.
        * `FeatureUnion([ ('pipeline_1', Pipeline([])), ('pipeline_2', TextLengthExtractor()), (...) ])`
      * How to use a feature union: It takes on all the methods of whatever the last estimator in its sequence is, which is the same as the general pipeline.
        * Example: `pipeline.fit(Xtrain)` first runs `features`'s estimator which is a feature union. In this feature union, `nlp_pipeline` and `txt_len` estimators run in parallel. Then these two estimators' results are combined and passed into the next estimator in the pipeline.

   3. Creating custom transformers 
      * By extending the base class in Scikit-Learn
         ```python
         import numpy as np
         from sklearn.base import BaseEstimator, TransformerMixin

         class TenMultiplier(BaseEstimator, TransformerMixin): 
             def __init__(self):
                 pass

             # To learn on the data
             def fit(self, X, y = None):
                 return self

             # To transform the data
             def transform(self, X): 
                 return X * 10
         ```
         * In custom transformer class, `fit` and `transform` method must exist to meet the requirement of an estimator.
         * `fit(X, y)`: This takes in a 2d array X for the feature data and a 1d array y for the target labels. By simply returning `self`, it allows us to chain methods together, since the result on calling fit on the transformer is still the transformer object. This method is required to be compatible with scikit-learn.
         * `transform(X)`: This is where we include the code that transforms the data.

      * By using `FunctionTransformer` from scikit-learn's preprocessing module
        ```python
        from sklearn.preprocessing import FunctionTransformer
        ```
        * This allows you to wrap an existing function to become a transformer. This provides less flexibility, but is much simpler.
        * [Scikit-learn's user guide](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer)
         
 ### 3. Grid Search
   1. Grid search: A tool that allows you to define a “grid” of parameters, or a set of values to check. For all possible combinations of values, it scores each combination with cross validation, and uses the cross validation score to determine the parameters that produce the most optimal model.
      * Running grid search on your pipeline allows you to try many parameter values thoroughly and conveniently, for both your data transformations and estimators.
      * Running it on your whole pipeline helps you test multiple parameter combinations across your entire pipeline. It accounts for interactions among parameters not just in your model, but data preparation steps as well.

   2. Using pipeline with GridSearchCV
      ```python
      pipeline = Pipeline([
          ('scaler', StandardScaler()),
          ('clf', SVC())
      ])
      # Dictionary of parameters to search, (key: value) pairs
      parameters = {
          'scaler__with_mean': [True, False], 
          'clf__kernel': ['linear', 'rbf'], 
           'clf__C':[1, 10]
      }
      
      cv = GridSearchCV(pipeline, param_grid = parameters)
      cv.fit(X_train, y_train)
      y_pred = cv.predict(X_test)
      ```
      * `pipeline = Pipeline([('scaler', StandardScaler()), ('clf', SVC())])`
        * `scaler`: To prevent data leakage, or preventing the validation set having knowlege of the whole training set, scaler is included in the pipeline. This makes sure that scaling is done only on the training set, and not the validation set within each fold of cross validation.
      * `parameters = {'key': 'values'}`
        * `key`: Names of the parameters and 
        * `values`: A list of parameter values to check
      * `cv.fit(X_train, y_train)`: This will run cross validation on all different combinations of these parameters to find the best combination of parameters for the model.

## Project: Disaster Response Pipeline
> Goal: Building a ML pipeline to categorize emergency messages based on the needs communicated by the sender <br>
> Required skills: data pipelines, NLP pipelines, machine learning pipelines, supervised learning

