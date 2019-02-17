import os
from collections import Counter
import re
import numpy as np
from nltk.corpus import stopwords

directory = os.path.join("c:\\", "path")

def word_count(str):
    words = str.split()
    for word in words:
        if word in bag_of_words:
            bag_of_words[word] += 1
        else:
            bag_of_words[word] = 1

def process_training_data(path_to_files):
    dictionary = list()
    bag_of_words = list()
    count = 0

    for file in os.listdir(path_to_files):
        if file.endswith(".txt"):
            with open(path_to_files+file, encoding='utf8') as f:
                counted_words = re.sub("[^\w']", " ", f.read()).split()
                tokens = [word for word in counted_words]
                dictionary.append(counted_words)
                bag_of_words.append(Counter(tokens))
            # print(file)
            count += 1
        
    return dictionary, bag_of_words

def do_prediction(document, weights):
    score = 0.0
    
    for word, counts in document.items():
        score += counts * weights[word]
#        print(score)
        if (score >= 0.0):
            return positive_review 
        else:
            return negative_review


def calculate_weights(array_of_docs, weighting, target_label):
    w = 0;
    for document in array_of_docs:
        predicted_y = do_prediction(document, weighting)
        if predicted_y != target_label:
            if predicted_y == positive_review:
                for word, counts in document.items():
                    weighting[word] += counts
            else:
                for word, counts in document.items():
                    weighting[word] -= counts

    return weighting

# Declaring variables puto!
positive_review = 1
negative_review = 0

dictionary1, bag_of_words1 = process_training_data('review_polarity/txt_sentoken/neg/')
dictionary2, bag_of_words2 = process_training_data('review_polarity/txt_sentoken/pos/')

# combined positive and negative dictionaries into one
#dictionary = dictionary1 + dictionary2


# create a dictionary with all the weights set to 0
weighting = {}
for word in dictionary1:
    weighting[word] = 0
    
weighting = calculate_weights(bag_of_words1[0:800], weighting, negative_review)
#weighting = calculate_weights(bag_of_words2[0:800], weighting, positive_review)

for document in bag_of_words1[800:1000]:
    print(do_prediction(document, weighting))
    

#print(bag_of_words1[0])
#print(calculate_weights())