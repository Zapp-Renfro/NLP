import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import pickle

nltk.download('punkt')
nltk.download('stopwords')

stop_words = list(stopwords.words('french'))
stemmer = FrenchStemmer()

def custom_tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if token.isalpha()]
    stems = [stemmer.stem(token) for token in filtered_tokens]
    return stems

def prepare_features(df, task, train_mode=True, vectorizer_path=None):
    if train_mode:
        labels = get_output(df, task)
    else:
        labels = None  # No target labels in non-training mode

    if train_mode or vectorizer_path is None:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=200000,
            stop_words=stop_words,
            use_idf=True,
            tokenizer=custom_tokenize_and_stem,
            ngram_range=(1, 3)
        )
        features = tfidf_vectorizer.fit_transform(df['video_name'])

        # Save the vectorizer to a file
        with open(vectorizer_path if vectorizer_path else 'custom_tfidf_vectorizer.pkl', 'wb') as file:
            pickle.dump(tfidf_vectorizer, file)
    else:
        # Load the vectorizer for predictions
        with open(vectorizer_path, 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        features = tfidf_vectorizer.transform(df['video_name'])

    if train_mode:
        return features, labels
    else:
        return features

def make_features(df, task, train_mode=True, vectorizer_path=None):
    if train_mode:
        y = get_output(df, task)
    else:
        y = None  # No target column in non-training mode

    if train_mode or vectorizer_path is None:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=200000,
            stop_words=stop_words,
            use_idf=True,
            tokenizer=custom_tokenize_and_stem,
            ngram_range=(1, 3)
        )
        X = tfidf_vectorizer.fit_transform(df['video_name'])

        # Save the vectorizer
        with open(vectorizer_path if vectorizer_path else 'custom_tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
    else:
        # Load the vectorizer for prediction
        with open(vectorizer_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        X = tfidf_vectorizer.transform(df['video_name'])

    if train_mode:
        return X, y
    else:
        return X

def get_output(df, task):
    if task == "is_humorous_video":
        return df["is_humorous"]
    elif task == "is_title":
        return df["is_title"]
    elif task == "find_humor_category":
        return df["humor_category"]
    else:
        raise ValueError("Unknown task")
