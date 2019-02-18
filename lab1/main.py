import os
from collections import Counter
import re
import random
from numpy import array, dot, random
import matplotlib.pyplot as plt


directory = os.path.join("c:\\", "path")


class Nlp:
    def __init__(self):
        self.training_documents = list()
        self.test_documents = list()
        self.weights = {}
        self.sum_of_weights = {}
        self.zeroed_weights = {}
        self.dictionary = Counter()
        self.vector_of_weights = [] 
        self.errors = 0
        self.errors_per_iteration = list()
        self.correct_predictions = 0
        self.positive_label = 1.0
        self.negative_label = -1.0
        self.seed = 20
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.debug = False

    """
    This function takes in a path to a directory and the label for
    the documents in that directory. Then using those documentas 
    it adds tuples containing (Document word counts, Label) 
    to the training_documents list.
    """

    def process_training_data(self, path_to_files, label):
        #        documents = list()
        for file in os.listdir(path_to_files)[:800]:
            with open(path_to_files + file, 'r') as f:
                counted_words = re.sub("[^\w']", " ", f.read()).split()
                dictionary = Counter(counted_words)
                self.dictionary += dictionary
                training_element = (dictionary, label)
                self.training_documents.append(training_element)

        for key, count in self.dictionary.items():
            self.weights[key] = 0
            self.zeroed_weights[key] = 0

    def process_test_data(self, path_to_files, label):
        for file in os.listdir(path_to_files)[800:]:
            with open(path_to_files + file, 'r') as f:
                counted_words = re.sub("[^\w']", " ", f.read()).split()
                dictionary = Counter(counted_words)
                test_data = (dictionary, label)
                self.test_documents.append(test_data)

    def calculate_weights_bag_of_words(self, shuffle):
        errors = 0
        count = 0
        random.seed(self.seed)
        if shuffle:
            random.shuffle(self.training_documents)
        for document, label in self.training_documents:
            count += 1
            predicted_label = self.predict_labels(document, self.weights)
            if predicted_label != label:
                errors += 1
                for word, counts in document.items():
                    self.weights[word] = self.weights[word] + (label * counts)
            self.errors_per_iteration.append(1 - (errors / count))

        if self.debug:
            print("------------------------")
            print("total errors", errors)
            print("accuracy %", 1 - (errors / len(self.training_documents)))
            print("doc count", len(self.training_documents))
            print("------------------------")

    """
    This function calculates the weights of the words found
    in the training_data list
    """

    def calculate_weights(self, iterations, shuffle):
        random.seed(self.seed)
        for i in range(iterations):
            errors = 0
            self.correct_predictions = 0
            if shuffle:
                random.shuffle(self.training_documents)
            for document, label in self.training_documents:
                predicted_label = self.predict_labels(document, self.weights)
                #                print(label, pred_lable)
                # self.record_results(label, predicted_label)
                if predicted_label != label:
                    # print(predicted_label, label)
                    errors += 1
                    for word, counts in document.items():
                        # print("word", word, "weight ", self.weights[word], " new weight ", self.weights[word] + label * counts, "predicted_label", predicted_label, "label", label)
                        self.weights[word] = self.weights[word] + label * counts
            if self.debug:
                print("------------------------")
                print("iteration", i + 1)
                print("total errors", errors)
                print("accuracy %", 1 - (errors / len(self.training_documents)))
                print("doc count", len(self.training_documents))
                print("------------------------")
            self.errors_per_iteration.append(1 - (errors / len(self.training_documents)))

    """
    Takes in a number of iterations for processing the data and a "debug" boolean
    to output to the console data about every iteration
    """
    def calculate_weights_averaged(self, iterations):
        random.seed(self.seed)
        c = 1  # used to keep track of which columns is being edited per iteration
        self.vector_of_weights.append(self.zeroed_weights.copy())
        self.sum_of_weights = self.zeroed_weights.copy()
        for i in range(iterations):
            errors = 0
            random.shuffle(self.training_documents)
            self.vector_of_weights.append(self.vector_of_weights[i].copy())

            for document, label in self.training_documents:

                # predicted_label = self.predict_labels(document, self.vector_of_weights[c-1])
                predicted_label = self.predict_labels(document, self.weights)
                if predicted_label != label:
                    errors += 1

                    for word, counts in document.items():
                        # self.vector_of_weights[c][word] = self.vector_of_weights[c-1][word] + label * counts
                        self.sum_of_weights[word] = self.sum_of_weights[word] + label * counts
                else:
                    for word, counts in document.items():
                        # self.vector_of_weights[c][word] = self.vector_of_weights[c-1][word]
                        self.sum_of_weights[word] = self.sum_of_weights[word]
                self.calculate_average_weights(c)
                c += 1

            if self.debug:
                print("------------------------")
                print("iteration", i + 1)
                print("total errors", errors)
                print("accuracy %", 1- (errors/len(self.training_documents)))
                print("doc count", self.training_documents)
                print("------------------------")
            self.errors_per_iteration.append(1 - (errors/len(self.training_documents)))
            # self.calculate_average_weights(c)

        # self.calculate_average_weights(c)
            # self.print_evaluation()
            # self.reset_evaluation()
            
    def calculate_average_weights(self, iterations):
        # print(self.weights)
        ## play with these functions, dont know which one is coreect
        # for word, weight in self.vector_of_weights[iterations-1].items():
        #     self.weights[word] += weight
        # for i in range(iterations-1):
        #     for word, weight in self.vector_of_weights[i].items():
        #         self.weights[word] += weight
        for word, weight in self.sum_of_weights.items():
            self.weights[word] = weight / iterations


    def plot_errors(self, title, xlabel, ylabel):
        plt.plot(self.errors_per_iteration)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        # plt.ylim([0, 1])
        plt.show()


    """
    This function returns a label for a given documents
    based on the words that occur in that same document
    """

    def predict_labels(self, document, weights):
        score = 0.0
        for word, counts in document.items():
            if word not in self.weights:
                self.weights[word] = 0
            score += counts * weights[word]
        if score >= 0.0:
            return 1.0
        else:
            return -1.0

    def evaluate_test_data(self):
        errors = 0
        for document, label in self.test_documents:
            # print(self.weights)
            predicted_label = self.predict_labels(document, self.weights)
            self.record_results(label, predicted_label)
            if predicted_label != label:
                # print(label, predicted_label)
                errors += 1

        print("------------------------")
        print("total errors", errors)
        print("accuracy %", 1- (errors / 400))
        print("total documents %", len(self.test_documents))
        print("------------------------")
        # print("TPos {}, FPos {}, TNeg {}, FNeg{}".format(self.true_positives, self.false_positives, self.true_negatives,
        #                                                  self.false_negatives))

    def record_results(self, actual_label, predicted_label):
        # print(actual_label, predicted_label)
        if actual_label == predicted_label:
            if actual_label == self.positive_label:
                # print("truepos")
                self.true_positives += 1
            else:
                # print("trueneg")
                self.true_negatives += 1
        else:
            if predicted_label == self.positive_label:
                # print("falsepos")
                self.false_positives += 1
            else:
                # print("falseneg")
                self.false_negatives += 1

    """
    Perfoms an analysis based on the classified documents
    and prints them in a human-readable way
    """

    def print_evaluation(self):
        print("TPos {}, FPos {}, TNeg {}, FNeg{}".format(self.true_positives, self.false_positives, self.true_negatives,
                                                         self.false_negatives))
        accuracy = (self.true_positives + self.true_negatives) / (
                    self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)
        print("Accuracy: {}".format(accuracy))
        precision = self.true_positives / (self.true_positives + self.false_positives)
        print("Precision: {}".format(precision))
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        print("Recall: {}".format(recall))
        f1_score = 2 * ((precision * recall) / (precision + recall))
        print("f1-score: {}".format(f1_score))


# labels
positive_label = 1.0
negative_label = -1.0

"""
Process data to use in NLP object
"""
def process_data(nlp):
    nlp.process_training_data('review_polarity/txt_sentoken/pos/', positive_label)
    nlp.process_training_data('review_polarity/txt_sentoken/neg/', negative_label)

    nlp.process_test_data('review_polarity/txt_sentoken/pos/', positive_label)
    nlp.process_test_data('review_polarity/txt_sentoken/neg/', negative_label)


## -----------------------------
## First perceptron
## -----------------------------
print("Exercise 1 a")
print("------------------------")
nlp = Nlp()
nlp.debug = True
nlp.process_training_data('review_polarity/txt_sentoken/neg/', negative_label)

nlp.process_training_data('review_polarity/txt_sentoken/pos/', positive_label)

nlp.process_test_data('review_polarity/txt_sentoken/pos/', positive_label)
nlp.process_test_data('review_polarity/txt_sentoken/neg/', negative_label)
# 1 repetition without shuffling the training data
nlp.calculate_weights_bag_of_words(True)

# n repetitions and shuffling the training data
# nlp.calculate_weights(50, True)

# nlp.calculate_weights_averaged(22) # accuracy .61
# nlp.calculate_weights_averaged(20) # accuracy .59
print("------------------------")
print("Plot Learning Curve")
title = "Accuracy Per Iteration"
ylabel = 'Accuracy'
xlabel = 'Documents'
nlp.plot_errors(title,xlabel,ylabel)

print("Evaluate data")
nlp.evaluate_test_data()
nlp.print_evaluation()




print("------------------------")
print("Perceptron 2")
nlp2 = Nlp()
nlp2.debug = True

nlp2.process_training_data('review_polarity/txt_sentoken/pos/', positive_label)
nlp2.process_training_data('review_polarity/txt_sentoken/neg/', negative_label)

nlp2.process_test_data('review_polarity/txt_sentoken/pos/', positive_label)
nlp2.process_test_data('review_polarity/txt_sentoken/neg/', negative_label)
# 1 repetition without shuffling the training data
# nlp2.calculate_weights(1, False)

# n repetitions and shuffling the training data
nlp2.calculate_weights(23, True)

# nlp.calculate_weights_averaged(22) # accuracy .61
# nlp2.calculate_weights_averaged(20) # accuracy .59
print("------------------------")
print("Plot Learning Curve")
title = "Accuracy Per Iteration"
ylabel = 'Accuracy'
xlabel = 'Iterations'
nlp2.plot_errors(title,xlabel,ylabel)

print("------------------------")
print("Evaluate data")
nlp2.evaluate_test_data()
nlp2.print_evaluation()