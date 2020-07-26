#!/usr/bin/python

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tools.parse_out_email_text import parse_out_text

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

BASE_PATH = os.path.split(os.path.abspath(os.getcwd()))[0]

# use str.replace() to remove any instances of the words
WORDS_TO_REMOVE = ["sara", "shackleton", "chris", "germani"]


def process_text():
    from_sara = open(os.path.join(BASE_PATH, "text_learning/from_sara.txt"), "r")
    from_chris = open(os.path.join(BASE_PATH, "text_learning/from_chris.txt"), "r")

    from_data = []
    word_data = []

    for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
        for path in from_person:
            path = os.path.join(BASE_PATH, "resources/datasets/enron_mail_20150507/", path[:-1])
            # print(path)
            email = open(path, "r")

            # use parseOutText to extract the text from the opened email
            processed_text = parse_out_text(email)

            for word in WORDS_TO_REMOVE:
                processed_text = processed_text.replace(word, "")

            # append the text to word_data
            if processed_text != "":
                word_data.append(processed_text)

            # append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if from_person == "sara":
                from_data.append(0)
            else:
                from_data.append(1)

            email.close()

    print("word data: {}".format(word_data[152]))
    print("emails processed")
    from_sara.close()
    from_chris.close()

    pickle.dump(word_data, open(os.path.join(BASE_PATH, "text_learning/your_word_data.pkl"), "w"))
    pickle.dump(from_data, open(os.path.join(BASE_PATH,  "text_learning/your_email_authors.pkl"), "w"))

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit_transform(word_data)
    feature_words = vectorizer.get_feature_names()
    print("number of words:", len(feature_words))
    print("word number 34597:", feature_words[34597])


if __name__ == "__main__":
    process_text()
