#!/usr/bin/python

import os
import pickle
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


BASE_PATH = os.path.split(os.path.abspath(os.getcwd()))[0]

numpy.random.seed(42)


def predict():

    # The words (features) and authors (labels), already largely processed.
    # These files should have been created from the previous (Lesson 10)
    # mini-project.
    words_file = os.path.join(BASE_PATH, "text_learning/your_word_data.pkl")
    authors_file = os.path.join(BASE_PATH, "text_learning/your_email_authors.pkl")

    word_data = pickle.load(open(words_file, "r"))
    authors = pickle.load(open(authors_file, "r"))

    # test_size is the percentage of events assigned to the test set (the
    # remainder go into training)
    # feature matrices changed to dense representations for compatibility with
    # classifier functions in versions 0.15.2 and earlier
    features_train, features_test, labels_train, labels_test = train_test_split(
        word_data, authors, test_size=0.1, random_state=42
    )

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english")
    features_train = vectorizer.fit_transform(features_train)
    features_test = vectorizer.transform(features_test).toarray()

    # a classic way to overfit is to use a small number
    # of data points and a large number of features;
    # train on only 150 events to put ourselves in this regime
    features_train = features_train[:150].toarray()
    labels_train = labels_train[:150]

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(features_train, labels_train)
    prediction = decision_tree.predict(features_test)
    important_features = decision_tree.feature_importances_

    print("accuracy: {}".format(accuracy_score(labels_test, prediction)))

    for i in important_features:
        if i >= 0.2:
            print("Important feature: ", i, " ", numpy.where(important_features == i))


if __name__ == "__main__":
    predict()
