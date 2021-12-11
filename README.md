# FRE7773 FINAL PROJECT

This project uses Bitcoin Tweets data (source: https://www.kaggle.com/kaushiksuresh147/bitcoin-tweets/tasks?taskId=3483) to do sentiment analysis on tweets text and utilizes tfidf vectorizer to convert a collection of raw documents to a matrix of TF-IDF features, then test multiple classifiers (SGDClassifier, MultinomialNB, Randomforestclassifier) to predict the sentiment for tweets based on the training data. All the experiments are tracked with Comet, and each logical components are isolated in Metaflow steps and important artifacts saved and versioned.

Last update: December 2021.

## Prequisites: Dependencies

requirements.txt file contains all the required packages, recommend using virtualenv to keep environments isolated, i.e. creating a new environment:

'python3 -m venv venv'

then activating it and installing the required dependencies:

'source venv/bin/activate'

'pip install -r requirements.txt'

## Project Structure

* meta_flow.py: this is the main file, including Bitcoin tweets data cleaning, sentiment analysis using textblob, text convertion using tfidf vectorizer, sentiment prediction using RandomForestClassifier, SGDClassifier and MultinomialNB, quantitative and qualitative test on model performance, vectorizer and model pickling for the Flask app to work

* best_model.pkl: the classifier model is pickled in this file, used for the Flask app

* best_vectorizer.pkl: the vectorizer is pickled in this file, used for the Flask app

* index.html: used to display the predicted result, needed to be placed in the templates file

## Acknowledgments
Thanks Prof. Meninder Purewal and Prof. Jacopo Tagliabue for their teach in ML and NLP and the great help and guidances in this project!
