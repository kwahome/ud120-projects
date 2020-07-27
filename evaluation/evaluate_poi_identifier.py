#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from tools.feature_format import feature_format, target_feature_split


BASE_PATH = os.path.split(os.path.abspath(os.getcwd()))[0]


def poi_identifier():
    data_dict = pickle.load(open(os.path.join(BASE_PATH, "final_project/final_project_dataset.pkl"), "r"))

    # add more features to features_list!
    features_list = ["poi", "salary"]

    data = feature_format(data_dict, features_list)
    labels, features = target_feature_split(data)

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.30, random_state=42
    )

    # Decision tree
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    prediction = clf.predict(features_test)
    print("accuracy:", accuracy_score(labels_test, prediction))

    # evaluation
    values, counts = np.unique(prediction, return_counts=True)
    test_size = len(features_test)
    print("predicted POIs:", zip(values, counts))
    print("total no in test set:", test_size)
    print("accuracy if all poi=0:", float(counts[0]) / float(test_size))

    true_positives = 0
    for actual, predicted in zip(labels_test, prediction):
        if actual == 1 and predicted == 1:
            true_positives += 1

    print("true positives:", true_positives)
    print("precision score:", precision_score(labels_test, prediction))
    print("recall score:", recall_score(labels_test, prediction))

    prediction_labels = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

    calculate_precision_and_recall(true_labels, prediction_labels)


def calculate_precision_and_recall(actual, predicted):
    print("Calculating precision and recall...")
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    for a, p in zip(actual, predicted):
        if a == 1 and p == 1:
            true_positives += 1
        elif a == 1 and p == 0:
            false_negatives += 1
        elif a == 0 and p == 1:
            false_positives += 1
        else:
            true_negatives += 1
    print("true positives:", true_positives)
    print("false positives:", false_positives)
    print("true negatives:", true_negatives)
    print("false negatives:", false_negatives)
    print("precision:", float(true_positives) / float(true_positives + false_positives))
    print("recall:", float(true_positives) / float(true_positives + false_negatives))


if __name__ == "__main__":
    poi_identifier()
