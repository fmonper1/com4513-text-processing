import os, sys, re, random, argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

class bigram_nlp:
    def __init__(self, bool_unigram, questions_uri, corpus_uri):
        self.training_documents = list(list())
        self.corpus_file = corpus_uri
        self.questions_file = questions_uri
        self.unigram_frequencies = dict()
        self.bigram_frequencies = dict()
        self.corpus_length = 0
        self.bigram_corpus_length = 0
        self.unique_words = 0
        self.sentences = list()
        self.unigram = bool_unigram

    """
    Opnes the corpus file and extracts the sentences and counts
    bigrams and unigrams
    """
    def process_training_data(self):
        with open(self.corpus_file) as f:
            for line in f:
                line_without_fullstop = line[:-3].lower()
                padded_line = SENTENCE_START+" "+line_without_fullstop+" "+SENTENCE_END
                padded_line = padded_line.lower()
                self.count_unigrams(padded_line)
                self.count_bigrams(padded_line)

    """
    
    """
    def count_unigrams(self, sentence):
        separate_words = sentence.split()
        for word in separate_words:
            if self.unigram_frequencies.get(word, 0) == 0: self.unique_words +=1
            self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
            if word != SENTENCE_START and word != SENTENCE_END:
                self.corpus_length += 1
        self.training_documents.append(separate_words)

    """
    Counts the occurency of bigrams in the training data
    """
    def count_bigrams(self, sentence):
        separate_words = self.get_bigram(sentence)
        # print(separate_words)
        for word1, word2 in separate_words:
            self.bigram_frequencies[(word1, word2)] = self.bigram_frequencies.get((word1, word2), 0) + 1
            self.bigram_corpus_length += 1
        # self.training_documents.append(separate_words)

    """
    Splits a sentence into a lsit of bigrams
    """
    def get_bigram(self, sentence):
        result = []
        arr = sentence.split()
        for i in range(len(arr) - 1):
            result.append((arr[i], arr[i + 1]))
        return result

    """
    Gets the sentences from the questions file
    and removes the candidate words and punctuation
    from it
    """
    def prepare_questions(self):
        sentences = list()
        with open(self.questions_file) as f:
            for line in f:
                separate_words = line.lower().split()
                possible_words = separate_words[-1]
                separate_words = separate_words[:-3] # remove the punctuation and words from the question
                separate_words.append(SENTENCE_END)
                separate_words.insert(0, SENTENCE_START)
                possible_words = possible_words.split("/")
                sentences.append((separate_words, possible_words))
        self.sentences = sentences
        return sentences

    """
    Calculates the probabilities of all sentence with the
    candidate words, it also prints out the value of the 
    probabilities
    """
    def calculate_questions(self, smoothing):
        for sentence, words in self.sentences:
            print("Sentence -----------")
            print(sentence, words)
            index = sentence.index("____")
            # print(index)
            sentence[index] = words[0]
            using_word0 = self.process_sentence(sentence, smoothing)
            sentence[index] = words[1]
            using_word1 = self.process_sentence(sentence, smoothing)
            if using_word0 > using_word1:
                print("Predicted: ", words[0], ": ", using_word0)
                print("Left over: ", words[1], ": ", using_word1)
            else:
                print("Predicted: ", words[1], ": ", using_word1)
                print("Left over: ", words[0], ": ", using_word0)

    """
    Thsi is the function that actually handles calculating
    the probability of sentences
    """
    def process_sentence(self, sentence, smoothing):
        probability = 1
        if self.unigram:
            for word in sentence:
                if word != SENTENCE_START and word != SENTENCE_END:
                    probability = probability * (self.unigram_frequencies.get(word, 0) / self.corpus_length)
            return probability
        else:
            c = 0
            for word in sentence:
                if word != SENTENCE_START and word != SENTENCE_END:
                    previous_word = sentence[c-1]
                    prob_word_and_previous = self.bigram_frequencies.get((previous_word, word), 0) + (1 * smoothing)
                    prob_previous = self.unigram_frequencies.get(previous_word, 0)  + (self.unique_words * smoothing)
                    try:
                        probability = probability * (prob_word_and_previous / prob_previous)
                    except ZeroDivisionError:
                        probability =  probability * 0

                    # print("(", previous_word, ", ", word, "): ", prob_word_and_previous)
                    # print("prob b", prob_previous)

                c += 1
            return probability

    """
    THis function checks that the total frequencies is the same
    as the corpus length for bigrams and unigrams
    """
    def test_correct(self):
        sum = 0
        for word in self.unigram_frequencies:
            if word != SENTENCE_START and word != SENTENCE_END:
                sum += self.unigram_frequencies[word]
        print("Unigram Corpus Length Ratio: ", sum / self.corpus_length)

        sum = 0
        for word in self.bigram_frequencies:
            if word != SENTENCE_START and word != SENTENCE_END:
                sum += self.bigram_frequencies[word]
        print("Bigram Corpus Length Ratio: ",sum / self.bigram_corpus_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('corpus')
    parser.add_argument('questions')

    args = parser.parse_args()
    questions_uri = args.questions
    corpus_uri = args.corpus
    print(questions_uri, corpus_uri)

    print("------ UNIGRAM NLM ------")
    unigram_nlp = bigram_nlp(True, questions_uri, corpus_uri)
    unigram_nlp.process_training_data()
    # print(len(unigram_nlp.training_documents))
    print("Unique words:", unigram_nlp.unique_words)
    print("Corpus length:", unigram_nlp.corpus_length)
    print(unigram_nlp.test_correct())
    unigram_nlp.prepare_questions()
    unigram_nlp.calculate_questions(0)


    print("------ NO SMOOTHING BIGRAM NLM ------")
    bigram_nlp = bigram_nlp(False, questions_uri, corpus_uri)
    bigram_nlp.process_training_data()
    print("Unique words:", bigram_nlp.unique_words)
    print("Corpus length:", bigram_nlp.corpus_length)
    print("Bigram Corpus length:", bigram_nlp.bigram_corpus_length)
    print(bigram_nlp.test_correct())
    bigram_nlp.prepare_questions()
    bigram_nlp.calculate_questions(0) # smoothing amount


    print("------ SMOOTHED BIGRAM NLM ------")
    bigram_nlp.prepare_questions()
    bigram_nlp.calculate_questions(.5) # smoothing amount