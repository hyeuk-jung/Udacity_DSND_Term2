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
      * Stemming: A process of reducing a word to its stem or root form. It often returns incomplete words.
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
        - tf(t, d): Term frequency, $count(t, d) / |d|$ (a term t in a document d / total number of term in document d)
        -  idf(t, D): Inverse document frequency, $\log(|D| / |{d \in D : t \in d}|)$ (log(D / the number of documents where term t is present))
      * Usage: Assigns weights to words that signify their relevance in documents

   3. Word Embeddings: Using a vector space, find an embedding for each word. It also conveys relationships among other words. This is useful when using one-hot encoding or other ways of numerical transformation is inefficient.
      * Usage: Finding synonyms and analogies, identifying concepts around which words are clustered, classifying words as positive, negative, and neutral, etc..
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


 ### 2. Feature Union


 ### 3. Grid Search


## Project: Disaster Response Pipeline
> Goal: Building a ML pipeline to categorize emergency messages based on the needs communicated by the sender <br>
> Required skills: data pipelines, NLP pipelines, machine learning pipelines, supervised learning

