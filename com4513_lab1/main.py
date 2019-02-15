import os
import collections as c
import re
import numpy as np

directory = os.path.join("c:\\", "path")

def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_(x): return x * (1 - x)        

def word_count(str):
    words = str.split()
    for word in words:
        if word in bag_of_words:
            bag_of_words[word] += 1
        else:
            bag_of_words[word] = 1

def process_training_data():
    bag_of_words = dict()
    count = 0

    for file in os.listdir("review_polarity/txt_sentoken/neg"):
        if file.endswith(".txt"):
            if count >= 800:
                # only use the first 800 documents for training
                break
            else:
                with open('review_polarity/txt_sentoken/neg/'+file, encoding='utf8') as f:
                    counted_words = re.sub("[^\w']", " ", f.read()).split()
                    bag_of_words.update(c.Counter(counted_words))
                # print(file)
                count += 1
        
    return bag_of_words


def calculate_weights(training_set):
    weighting = dict()
    w = 0;
    for key, count in training_set.items():
        if key not in weighting:
            weighting[key] = 0
        pred_y = w * count
        if count != pred_y:
            if pred_y == 1:
                weighting[key] += training_set[key]
            else:
                weighting[key] -= training_set[key]
    return weighting

def prediction():
    return

bag_o_words = process_training_data()
calculated_weight = calculate_weights(bag_o_words)
print(bag_o_words)
print(calculated_weights)