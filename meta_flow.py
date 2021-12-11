from comet_ml import Experiment
from metaflow import FlowSpec, step, Parameter, current
from datetime import datetime
import pandas as pd
import os
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.metrics import confusion_matrix, classification_report

os.environ['COMET_API_KEY'] = 'Db1PkZ7B5XG5Mcv7ugvV80E5R'
os.environ['MY_PROJECT_NAME'] = 'ML7773_final_project'

# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'

class Bitcoin_Tweet_Flow(FlowSpec):
    """
    Bitcoin_Tweet_Flow is a minimal DAG showcasing a scikit text classification pipeline for sentiment
    analysis over Bitcoin tweets.
    Data from: https://www.kaggle.com/kaushiksuresh147/bitcoin-tweets/code
    """

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20)

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: {}".format(current.flow_name))
        print("run id: {}".format(current.run_id))
        print("username: {}".format(current.username))

        self.next(self.load_data)

    @step
    def load_data(self):
        """
        load the Bitcoin_tweets data in pickle file
        """

        # get the dataset and use self to version it
        self.data_ori = pd.read_pickle(r'Bitcoin_tweets.pkl')
        # drop the rows where the "text" and 'hashtags' columns are NaN
        self.data_ori = self.data_ori.dropna(subset=['text', 'hashtags']).reset_index(drop=True)
        # get tweets data only
        self.text_data = self.data_ori[['text']].copy()
        self.text_data.columns = ['tweets']
        # debug / info
        print("Total # of sentences loaded is: {}".format(len(self.text_data)))
        # go to the next step
        self.next(self.clean_data)

    @step
    def clean_data(self):
        # create a function to clean the tweets
        def tweets_clean(data):
            # I.
            # 1. Remove urls/hyperlinks
            tweet_without_url = re.sub(r'((www\.[^\s]+)|(http\S+))', ' ', data)
            # 2. Remove hashtags
            tweet_without_hashtag = re.sub(r'#\w+', ' ', tweet_without_url)
            # 3. Remove mentions
            tweet_without_mentions = re.sub(r'@\w+', ' ', tweet_without_hashtag)
            # 4. Remove characters that not in the English alphabets
            tweet_pre_cleaned = re.sub('[^A-Za-z]+', ' ', tweet_without_mentions)
            # 5. Remove additional white spaces
            tweet_pre_cleaned = re.sub('[\s]+', ' ', tweet_pre_cleaned)
            # II.
            # 1. Tokenize
            tweet_tokens = TweetTokenizer().tokenize(tweet_pre_cleaned)
            # 2. Lower
            tweet_lower = [l.lower() for l in tweet_tokens]
            # 3. Remove Puncs
            tokens_without_punc = [w for w in tweet_lower if w.isalpha()]
            # 4. Removing Stopwords
            lemmatizer = WordNetLemmatizer()
            # Import nltk stopwords and customize it to add common crypto words that don't add too much information
            stop_words = stopwords.words(['english'])
            # crypto_words = ['btc', 'bitcoin', 'eth', 'etherum', 'crypto']
            # stop_words = stop_words + crypto_words
            tokens_without_stopwords = [t for t in tokens_without_punc if t not in stop_words]
            # 5. lemmatize
            text_cleaned = [lemmatizer.lemmatize(t) for t in tokens_without_stopwords]
            # 6. Joining
            return " ".join(text_cleaned)

        # create a column to store the clean tweets
        self.text_data['cleaned_tweets'] = self.text_data['tweets'].apply(tweets_clean)
        # go to the next step
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        """
        Check data for anomalous data points and weird labels
        """
        # first, check all sentences are "long enough", > 20 chars, otherwise flag them
        for s in self.text_data['cleaned_tweets']:
            if len(s) < 5:
                print("====> Sentence '{}' seems too short, delete it for now".format(s))
                continue
            
        self.next(self.sentiment_analysis)

    @step
    def sentiment_analysis(self):
        """
        using textblob to do sentiment analysis
        """
        from textblob import TextBlob
        # create a function to get subjectivity
        # subjectivity is in range [0,1]
        def getSubjectivity(tweet):
            return TextBlob(tweet).sentiment.subjectivity
        # create a function to get the polarity
        # polarity is in range [-1,1]
        def getPolarity(tweet):
            return TextBlob(tweet).sentiment.polarity
        # create a function to get sentiment text based on polarity
        def getSentiment(score):
            if score < 0:
                return 'negative'
            elif score == 0:
                return 'neutral'
            else:
                return 'positive'

        # create two new columns called "Subjectivity" & "Polarity"
        self.text_data['subjectivity'] = self.text_data['cleaned_tweets'].apply(getSubjectivity)
        self.text_data['polarity'] = self.text_data['cleaned_tweets'].apply(getPolarity)
        # create a column to store the text sentiment
        self.text_data['sentiment'] = self.text_data['polarity'].apply(getSentiment)

        # if data is all good, let's go to training
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        """
        Train / test split
        """
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.text_data['cleaned_tweets'],
            self.text_data['sentiment'],
            test_size=self.TEST_SPLIT,
            random_state=42)

        # debug / info
        print("# train sentences: {},  # test: {}".format(len(self.X_train), len(self.X_test)))

        self.next(self.prepare_features)

    @step
    def prepare_features(self):
        """
        Transform our Xs (the sentences) using TF-IDF,
        i.e. given a list of sentences, return a list of vectors based on TF-IDF weighting scheme
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(analyzer='word')
        self.X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vectorized = self.vectorizer.transform(self.X_test)

        # train multiple models now that we have the features
        self.next(self.train_classifier1,self.train_classifier2,self.train_classifier3)

    @step
    def train_classifier1(self):
        """
        train a MultinomialNB model on the vectorized text
        """
        from sklearn.naive_bayes import MultinomialNB
        model1 = MultinomialNB()
        model1.fit(self.X_train_vectorized, self.y_train)
        # versioned the trained model using self
        self.trained_model1 = model1
        # go to the testing phase
        self.next(self.join)

    @step
    def train_classifier2(self):
        """
        train a SGDClassifier model on the vectorized text
        """
        from sklearn.linear_model import SGDClassifier
        model2 = SGDClassifier(random_state=42)
        model2.fit(self.X_train_vectorized, self.y_train)
        # versioned the trained model using self
        self.trained_model2 = model2
        # go to the testing phase
        self.next(self.join)

    @step
    def train_classifier3(self):
        """
        train a RandomForestClassifier model on the vectorized text
        """
        from sklearn.ensemble import RandomForestClassifier
        model3 = RandomForestClassifier(random_state=42)
        model3.fit(self.X_train_vectorized, self.y_train)
        # versioned the trained model using self
        self.trained_model3 = model3
        # go to the testing phase
        self.next(self.join)

    @step
    def join(self, inputs):
        """
        test the three models quantitatively and find the best model
        """
        from sklearn.metrics import accuracy_score

        # test the models with accuracy score
        predicted1 = inputs.train_classifier1.trained_model1.predict(inputs.train_classifier1.X_test_vectorized)
        self.accuracy_score1 = accuracy_score(inputs.train_classifier1.y_test, predicted1)

        predicted2 = inputs.train_classifier2.trained_model2.predict(inputs.train_classifier2.X_test_vectorized)
        self.accuracy_score2 = accuracy_score(inputs.train_classifier2.y_test, predicted2)

        predicted3 = inputs.train_classifier3.trained_model3.predict(inputs.train_classifier3.X_test_vectorized)
        self.accuracy_score3 = accuracy_score(inputs.train_classifier3.y_test, predicted3)

        # print out the report
        print("Classification Report")
        print('MultinomialNB, accuracy score: {}'.format(self.accuracy_score1))
        print('SGDClassifier, accuracy score: {}'.format(self.accuracy_score2))
        print('RandomForestClassifier, accuracy score: {}'.format(self.accuracy_score3))
        accuracy_dic = {'MultinomialNB':[self.accuracy_score1,inputs.train_classifier1.trained_model1],
                        'SGDClassifier':[self.accuracy_score2,inputs.train_classifier2.trained_model2],
                        'RandomForestClassifier':[self.accuracy_score3,inputs.train_classifier3.trained_model3]}
        best_model_symbol = max(accuracy_dic, key=accuracy_dic.get)
        self.best_accuracy = accuracy_dic[best_model_symbol][0]
        self.best_model = accuracy_dic[best_model_symbol][1]
        self.x_Test = inputs.train_classifier1.X_test
        self.y_Test = inputs.train_classifier1.y_test
        self.y_predicted = self.best_model.predict(inputs.train_classifier1.X_test_vectorized)
        self.Vectorizer = inputs.train_classifier1.vectorizer
        print('the best model is {} with accuracy score: {}'.format(best_model_symbol,self.best_accuracy))

        # these will be logged to your project on Comet.ml
        # sends a summary of metrics to Comet in order to track them!
        exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'],
                         auto_param_logging=False)

        params = {'random_state': 42,
                  'vectorizer':'TfidfVectorizer',
                  'classifier 1':'MultinomialNB',
                  'classifier 2':'SGDClassifier',
                  'classifier 3':'RandomForestClassifier',
                  'best classifier model':best_model_symbol,
                  'stratify': True
                  }

        metrics = {'MultinomialNB': self.accuracy_score1,
                   'SGDClassifier': self.accuracy_score2,
                   'RandomForestClassifier': self.accuracy_score3
                   }

        exp.log_parameters(params)
        exp.log_metrics(metrics)

        def sentiment_to_number(sentiment_list):
            number_list = []
            for i in sentiment_list:
                if i == 'positive':
                    number_list.append(2)
                elif i == 'neutral':
                    number_list.append(1)
                else:
                    number_list.append(0)
            return number_list

        self.y_test_number = sentiment_to_number(inputs.train_classifier2.y_test)
        self.y_pred_number = sentiment_to_number(self.best_model.predict(inputs.train_classifier2.X_test_vectorized))
        exp.log_confusion_matrix(self.y_test_number,self.y_pred_number)
        
        # Logging-in debug
        print('Sucessfully logging into Comet.ml!')
        # All is done go to the end
        self.next(self.beahvioral_tests)

    @step
    def beahvioral_tests(self):
        """
        As we learned in the course, it is very important to not just test quantitave behavior, but diving
        deep into model performances on cases of interest, slices of data, perturbed input.continue
        Note: we don't make the Flow fail here, but just flag when a test does not have the desired result.
        Other choices are possible of course.
        """
        from random import randint
        # Report performances on certain words, ex: bear, bull,
        def test_on_words(x_test, predicted, y_test, target='bear'):
            target_golden = []
            target_predicted = []
            for x, p, y in zip(x_test, predicted, y_test):
                if target in x:
                    target_golden.append(y)
                    target_predicted.append(p)
            return classification_report(target_golden, target_predicted)
        print(test_on_words(self.x_Test,self.y_predicted,self.y_Test,target='bear'))

        def create_perturbated_sentences(sentences: list):
            """
                Use BackTranslation to perform back-translation.
            """

            from BackTranslation import BackTranslation

            trans = BackTranslation(url=[
                'translate.google.com',
                'translate.google.co.kr',
            ])

            translated_sentences = []
            for t in sentences:
                result = trans.translate(t, src='en', tmp='zh-cn')
                translated_sentences.append(result.result_text)

            assert len(translated_sentences) == len(sentences)

            return translated_sentences

        # finally, some perturbation tests over 2 randomly sampled cases
        rnd_index = [randint(0, len(self.x_Test)) for _ in range(5)]
        self.test_sentences = [list(self.x_Test)[_] for _ in rnd_index]
        self.test_predictions = [list(self.y_predicted)[_] for _ in rnd_index]
        self.perturbated_test_sentences = create_perturbated_sentences(self.test_sentences)
        # run  predictions on perturbated inputs and compare the output
        self.new_Xs = self.Vectorizer.transform(self.perturbated_test_sentences)
        self.new_Ys = self.best_model.predict(self.new_Xs)
        print("\n@@@@ Perturbation tests @@@@\n")
        for original, pert, pred, y in zip(self.test_sentences, self.perturbated_test_sentences, self.test_predictions,
                                           self.new_Ys):
            print("\n\nOriginal: '{}', Perturbated: '{}'\n".format(original, pert))
            print("Original Y: '{}', Perturbated Y: '{}'\n".format(pred, y))
            if y != pred:
                print("ATTENTION: label changed after perturbation!\n")

        # all is done, dump the model
        self.next(self.dump_for_serving)

    @step
    def dump_for_serving(self):
        """
        Make sure we pickled the artifacts necessary for the Flask app to work
        Hint: is there a better way of doing this than pickling feature prep and model in two files? ;-)
        """
        import pickle
        working_path = os.getcwd()
        vec_file_name = working_path+'/best_vectorizer.pkl'
        model_file_name = working_path+'/best_model.pkl'
        with open(vec_file_name,'wb') as f:
            pickle.dump(self.Vectorizer,f)

        with open(model_file_name,'wb') as f:
            pickle.dump(self.best_model,f)

        # go to the end
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))

if __name__ == '__main__':
    Bitcoin_Tweet_Flow()