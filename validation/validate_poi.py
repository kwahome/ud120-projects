#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tools.feature_format import feature_format, target_feature_split


BASE_PATH = os.path.split(os.path.abspath(os.getcwd()))[0]


def poi_identifier():
    data_dict = pickle.load(open(os.path.join(BASE_PATH, "final_project/final_project_dataset.pkl"), "r"))

    # first element is our labels, any added elements are predictor
    # features. Keep this the same for the mini-project, but you'll
    # have a different feature list when you do the final project.
    features_list = ["poi", "salary"]

    data = feature_format(data_dict, features_list)
    labels, features = target_feature_split(data)

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )

    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)

    prediction = clf.predict(features_test)

    print("accuracy: {}".format(accuracy_score(prediction, labels_test)))


if __name__ == "__main__":
    poi_identifier()
