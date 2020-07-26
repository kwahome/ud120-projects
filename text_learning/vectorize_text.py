#!/usr/bin/python

import os
import pickle
import re
import sys
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


def process_text():
    from_sara = open(os.path.join(BASE_PATH, "text_learning/from_sara.txt"), "r")
    from_chris = open(os.path.join(BASE_PATH, "text_learning/from_chris.txt"), "r")

    from_data = []
    word_data = []

    # temp_counter is a way to speed up the development--there are
    # thousands of emails from Sara and Chris, so running over all of them
    # can take a long time
    # temp_counter helps you only look at the first 200 emails in the list so you
    # can iterate your modifications quicker
    temp_counter = 0

    for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
        for path in from_person:
            # only look at first 200 emails when developing
            # once everything is working, remove this line to run over full dataset
            temp_counter += 1
            if temp_counter < 200:
                path = os.path.join(BASE_PATH, "resources/datasets/enron_mail_20150507/", path[:-1])
                print(path)
                email = open(path, "r")

                # use parseOutText to extract the text from the opened email
                processed_text = parse_out_text(email)
                print("processed text: {}".format(processed_text))

                # use str.replace() to remove any instances of the words
                to_remove = ["sara", "shackleton", "chris", "germani"]
                processed_text = " ".join(["" if word in to_remove else word for word in list(processed_text)])

                # append the text to word_data
                word_data.append(processed_text)

                # append a 0 to from_data if email is from Sara, and 1 if email is from Chris
                if from_person == "sara":
                    from_data.append(0)
                elif from_person == "chris":
                    from_data.append(1)

                email.close()

    print("word data: {}".format(word_data[152]))
    print("emails processed")
    from_sara.close()
    from_chris.close()

    pickle.dump(word_data, open(os.path.join(BASE_PATH, "text_learning/your_word_data.pkl"), "w"))
    pickle.dump(from_data, open(os.path.join(BASE_PATH,  "text_learning/your_email_authors.pkl"), "w"))

    ### in Part 4, do TfIdf vectorization here


if __name__ == "__main__":
    process_text()
