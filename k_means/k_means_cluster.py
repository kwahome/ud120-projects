#!/usr/bin/python

""" 
    Skeleton code for k-means clustering mini-project.
"""


import pickle
import numpy
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from tools.feature_format import feature_format, target_feature_split


BASE_PATH = os.path.split(os.path.abspath(os.getcwd()))[0]


def draw(
    pred,
    features,
    poi,
    mark_poi=False,
    name="image.png",
    f1_name="feature 1",
    f2_name="feature 2",
):
    """ some plotting code designed to help you visualize your clusters """

    # plot each cluster with a different color--add more colors for
    # drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    # if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()


def cluster():
    # load in the dict of dicts containing all the data on each person in the dataset
    data_dict = pickle.load(
        open(os.path.join(BASE_PATH, "final_project/final_project_dataset.pkl"), "r")
    )

    # there's an outlier--remove it!
    data_dict.pop("TOTAL", 0)

    # the input features we want to use
    # can be any key in the person-level dictionary (salary, director_fees, etc.)
    feature_1 = "salary"
    feature_2 = "exercised_stock_options"
    feature_3 = "total_payments"
    poi = "poi"
    features_list = [poi, feature_1, feature_2, feature_3]

    data = feature_format(data_dict, features_list)
    poi, finance_features = target_feature_split(data)

    # in the "clustering with 3 features" part of the mini-project,
    # you'll want to change this line to
    # for f1, f2, _ in finance_features:
    # (as it's currently written, the line below assumes 2 features)
    for f1, f2, _ in finance_features:
        plt.scatter(f1, f2)
    plt.show()

    # k_means = KMeans(n_clusters=2, random_state=0)
    # k_means.fit(finance_features)
    # pred = k_means.predict(finance_features)

    # rename the "name" parameter when you change the number of features
    # so that the figure gets saved to a different file
    try:
        draw(
            pred,
            finance_features,
            poi,
            mark_poi=False,
            name="clusters-3.pdf",
            f1_name=feature_1,
            f2_name=feature_2,
        )
    except NameError:
        print("no predictions object named pred found, no clusters to plot")


if __name__ == "__main__":
    cluster()
