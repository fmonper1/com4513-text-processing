import os
from collections import Counter
import re
import random
from numpy import array, dot, random

directory = os.path.join("c:\\", "path")


class Nlp:
    def __init__(self):
        self.training_documents = list()
        self.test_documents = list()
        self.test_documents_shuffled = list()
        self.weights = {}
        self.vector_of_weights = list({})
        self.errors = 0
        self.positive_lable = 1
        self.negative_lable = 0
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
     
    """
    This function takes in a path to a directory and the label for
    the focuments in that directory. Then using those documentas 
    it adds tuples containing (Document word counts, Label) 
    to the training_documents list.
    """
    def process_training_data(self, path_to_files, label):
#        documents = list()
        for file in os.listdir(path_to_files)[:800]:
            with open(path_to_files+file,'r') as f:
                counted_words = re.sub("[^\w']", " ", f.read()).split()
                for word in counted_words:
                    if(word not in self.weights):
                            self.weights[word] = 0
                            
                dictionary = Counter(counted_words)
                tuple = (dictionary, label)
#                documents.append(tuple)
                self.training_documents.append(tuple)
        
    """
    This function takes in the predicted label and the actual
    label of a document. Then based on a comparsion it determines
    is the documents has been classified as a true positive, true negative,
    false positive or false negative for analysis purposes
    """        
    def record_results(self, actual_lable, predicted_lable):
        if actual_lable is predicted_lable:
            if actual_lable is self.positive_lable:
                self.true_positives += 1
            else:
                self.true_negatives += 1
        else:
            if predicted_lable is self.positive_lable:
                self.false_positives += 1
            else:
                self.false_negatives += 1
                
    def reset_evaluation(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
    
    """
    Perfoms an analysis based on the classified documents
    and prints them in a human-readable way
    """
    def print_evaluation(self):
        print("TPos {}, FPos {}, TNeg {}, FNeg{}".format(self.true_positives,self.false_positives,self.true_negatives,self.false_negatives))
        accuracy = (self.true_positives+self.true_negatives) / (self.true_positives+self.true_negatives+self.false_positives+self.false_negatives)
        print("Accuracy: {}".format(accuracy))
        precision = self.true_positives / (self.true_positives + self.false_positives)
        print("Precision: {}".format(precision))
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        print("Recall: {}".format(recall))
        f1_score = 2 * ((precision * recall) / (precision + recall))
        print("f1-score: {}".format(f1_score))        
        
    """
    This function calculates the weights of the words found
    in the training_data list
    """
    def calculate_weights(self):
        w = 0;
        for document, label in self.training_documents:
            pred_lable = self.predict_lable(document)
#            print(label, pred_lable)
            self.record_results(label, pred_lable)
            if pred_lable is not label:
                self.errors += 1
                if label is self.positive_lable :
#                    print("positive")
#                    print(self.positive_lable)
                    for word, counts in document.items():
                        self.weights[word] += counts
                else:
#                    print("negative")
                    for word, counts in document.items():
                        self.weights[word] -= counts
                        
    """
    This function calculates the weights of the words found
    in the training_data list
    """
    def calculate_weights_with_shuffle(self):
        w = 0;
        random.seed(15)
        random.shuffle(self.training_documents)
        for document, label in self.training_documents:
            pred_lable = self.predict_lable(document)
#            print(label, pred_lable)
            self.record_results(label, pred_lable)
            if pred_lable is not label:
                self.errors += 1
                if label is self.positive_lable :
#                    print("positive")
#                    print(self.positive_lable)
                    for word, counts in document.items():
                        self.weights[word] += counts
                else:
#                    print("negative")
                    for word, counts in document.items():
                        self.weights[word] -= counts

    """
    This function calculates the weights of the words found
    in the training_data list
    """
    def calculate_weights_with_shuffle_and_iterations(self, iterations):
        w = 0
        c = 1 # used to keep track of which columns is being edited per iteration
        c_minus_one = 0
        weight_vector_column = {}
        self.vector_of_weights.append({})

        for i in range(iterations):
            random.seed(15)
            random.shuffle(self.training_documents)
            self.vector_of_weights.append({})
            print(self.vector_of_weights)
            print(self.vector_of_weights[c_minus_one])
            print(self.vector_of_weights[c])
            for document, label in self.training_documents:
                pred_lable = self.predict_label_using_vector(document, c_minus_one)
    #            print(label, pred_lable)
                self.record_results(label, pred_lable)
                if pred_lable is not label:
                    self.errors += 1
                    if label is self.positive_lable :
                        for word, counts in document.items():
                            self.vector_of_weights[c][word] = self.vector_of_weights[c_minus_one][word] + counts
                        for word, counts in document.items():
                            self.vector_of_weights[c][word] = self.vector_of_weights[c_minus_one][word] - counts
                else:
                    for word, counts in document.items():
                        self.vector_of_weights[c][word] = self.vector_of_weights[c_minus_one][word]
            c += 1 # increase the column index every iteration
            c_minus_one += 1  # increase the column index every iteration
        self.calculate_average_weights(c)

    def calculate_average_weights(self, columns):
        for i in range(columns):
            for word, weight in self.vector_of_weights[i]:
                self.weights[word] += weight
        for word, weight in self.weights:
            self.weights[word] = weight/columns
    
    """
    This funcition returns a label for a given documents
    based on the words that occur in that same document
    """                    
    def predict_lable(self, document):
        score = 0.0
        for word, counts in document.items():
            if word not in self.weights:
                self.weights[word] = 0
            score += counts * self.weights[word]
            if score >= 0.0:
                return self.positive_lable
            else:
                return self.negative_lable

    """
    This funcition returns a label for a given documents
    based on the words that occur in that same document
    """
    def predict_label_using_vector(self, document, column_index):
        score = 0.0
        for word, counts in document.items():
            if word not in self.vector_of_weights[column_index]:
                self.vector_of_weights[column_index][word] = 0
            score += counts * self.vector_of_weights[column_index][word]
            if score >= 0.0:
                return self.positive_lable
            else:
                return self.negative_lable

# labels
positive_label = 1
negative_label = 0

print("Basic perceptron")
nlp = Nlp()
nlp.process_training_data('review_polarity/txt_sentoken/pos/', positive_label)
nlp.process_training_data('review_polarity/txt_sentoken/neg/', negative_label)

nlp.calculate_weights()
nlp.print_evaluation()


print("Basic perceptron with randomized training data")
nlp2 = Nlp()
nlp2.process_training_data('review_polarity/txt_sentoken/pos/', positive_label)
nlp2.process_training_data('review_polarity/txt_sentoken/neg/', negative_label)

nlp2.calculate_weights_with_shuffle()
nlp2.print_evaluation()

print("Basic perceptron with randomized training data and iterations")
nlp3 = Nlp()
nlp3.process_training_data('review_polarity/txt_sentoken/pos/', positive_label)
nlp3.process_training_data('review_polarity/txt_sentoken/neg/', negative_label)

nlp3.calculate_weights_with_shuffle_and_iterations(25)
nlp3.print_evaluation()