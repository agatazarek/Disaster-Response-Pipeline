import sys
import nltk
import pandas as pd
import pickle
import sqlalchemy

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)

    category_names = df.columns[4:].values
    X = df["message"]
    Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)

    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('best', TruncatedSVD()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__learning_rate': [1, 2]
    }

    return GridSearchCV(pipeline, param_grid = parameters, cv = 3,)


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(category_names[i])
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        nltk.download('popular')

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
