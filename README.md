# NLP_Accident_Classifier

#   Industrial Safety and Health Accident Analysis and Chatbot

    This repository contains the code and analysis for the "Industrial Safety and Health Database with Accidents Description" dataset. The project is divided into two main parts:

    1.  **Data Analysis, Preprocessing and Conventional Modelling:** This part focuses on exploring the dataset, cleaning the text descriptions, and preparing the data for further analysis and XGBoost Modeling.
    2.  **NLP Chatbot Development:** This part builds upon the preprocessed data to develop a chatbot that can potentially provide insights or answer questions related to the accident descriptions which uses RNNs, and Tranformer.

##   Files

    * `NLP_Accident_Classification.ipynb`: This notebook covers the data analysis, cleaning, and preprocessing steps. It also focuses on building the NLP chatbot using various techniques.
    * `Data Set - Industrial Safety and Health Database with Accidents Description.xlsx`: The dataset used in the project.

##   Data Description

    The dataset contains information about industrial safety and health accidents, including:

    * **Date:** Date of the accident.
    * **Country:** Country where the accident occurred.
    * **Local:** Location of the accident.
    * **Industry Sector:** Sector of the industry.
    * **Accident Level:** Severity of the accident.
    * **Potential Accident Level:** Potential severity of the accident.
    * **Genre:** Gender of the employee involved.
    * **Employee or Third Party:** Whether the injured party was an employee or a third party.
    * **Critical Risk:** The most critical risk factor.
    * **Description:** A textual description of the accident.

##   Data Analysis and Preprocessing:

    The following steps were performed in the data analysis and preprocessing notebook:

    * **Exploratory Data Analysis (EDA):**
        * Missing value analysis.
        * Duplicate record removal.
        * Visualization of accident distribution by locality and country.
        * Analysis of accident level and gender.
        * Identification of top-3 critical risks for each accident level.
        * Time-based analysis of accident occurrences.
        
    * **Data Cleaning:**
        * Lowercasing text.
        * Handling contractions (e.g., "don't" to "do not").
        * Correcting misspellings.
        * Removing punctuation.
        * Removing numbers and special characters.
        
    * **NLP Preprocessing:**
        * Tokenization.
        * Stop word removal.
        * Lemmatization.
        
    * **Feature Engineering:**
        * Extraction of `Year`, `Month`, and `Day` features from the `Date` column.
        
    * **Text Vectorization:**
        * Conversion of text descriptions into numerical features using TF-IDF.
        
    * **Encoding Categorical Features:**
        * Conversion of categorical variables into numerical representations.

##   NLP Chatbot Development:

    * **Algorithms and Techniques:**
        * XGBoost using TF-IDF
        * Recurrent Neural Networks (RNNs) and Bidirectional RNNs (BiRNNs)
        * Transformer Models - MiniLM
        * Word Embeddings (e.g., Word2Vec, GloVe, Sentence Transformers)
        
    * **Hyperparameter Tuning:**
        * GridSearchCV
        * RandomizedSearchCV
        
    * **Model Evaluation:**
        * Accuracy
        * Classification Report
        * Confusion Matrix

##   Dependencies

    * Python 3.x
    * pandas
    * numpy
    * matplotlib
    * seaborn
    * scikit-learn
    * nltk
    * contractions
    * symspellpy
    * gensim
    * torch
    * SentenceTransformer
    * huggingface\_hub
    * xgboost

