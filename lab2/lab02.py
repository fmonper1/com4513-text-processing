import os, sys, re, random, argparse
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

class unigram_nlp:
    def __init__(self):
        self.training_documents = list(list())
        self.corpus_file = "./news-corpus-500k.txt"
        self.questions_file = "./questions.txt"
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        self.unique_words = 0
        self.sentences = list()

    def process_training_data(self):
        with open(self.corpus_file) as f:
            for line in f:
                separate_words = line.split()
                separate_words.pop()
                separate_words.append(SENTENCE_END)
                separate_words.insert(0, SENTENCE_START)
                for word in separate_words:
                    self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                    if word != SENTENCE_START and word != SENTENCE_END:
                        self.corpus_length += 1
                self.training_documents.append(separate_words)
        # subtract 2 because unigram_frequencies dictionary contains values for SENTENCE_START and SENTENCE_END
        self.unique_words = len(self.unigram_frequencies) - 2

    def process_sentence(self, sentence):
        probability = 1
        for word in sentence:
            if word != SENTENCE_START and word != SENTENCE_END:
                probability = probability * (self.unigram_frequencies.get(word, 0)/ self.corpus_length)
        return probability

    def prepare_questions(self):
        sentences = list()
        with open(self.questions_file) as f:
            for line in f:
                separate_words = line.split()
                possible_words = separate_words[-1]
                separate_words = separate_words[:-2]
                possible_words = possible_words.split("/")
                sentences.append((separate_words, possible_words))
        self.sentences = sentences
        return sentences

    def calculate_questions(self):
        for sentence, words in self.sentences:
            print(sentence, words)
            index = sentence.index("____")
            # print(index)
            sentence[index] = words[0]
            using_word0 = self.process_sentence(sentence)
            sentence[index] = words[1]
            using_word1 = self.process_sentence(sentence)
            if using_word0 > using_word1:
                print("correct: ", words[0], ": ", using_word0)
                print("false: ", words[1], ": ", using_word1)
            else:
                print("correct: ", words[1], ": ", using_word1)
                print("false: ", words[0], ": ", using_word0)

    def test_correct(self):
        sum = 0
        for word in self.unigram_frequencies:
            if word != SENTENCE_START and word != SENTENCE_END:
                sum += self.unigram_frequencies[word]
        print(sum / self.corpus_length)


class bigram_nlp:
    def __init__(self, bool_unigram):
        self.training_documents = list(list())
        self.corpus_file = "./news-corpus-500k.txt"
        self.questions_file = "./questions.txt"
        self.unigram_frequencies = dict()
        self.bigram_frequencies = dict()
        self.corpus_length = 0
        self.bigram_corpus_length = 0
        self.unique_words = 0
        self.sentences = list()
        self.unigram = bool_unigram

    def process_training_data(self):
        with open(self.corpus_file) as f:
            for line in f:
                line_without_fullstop = line[:-3].lower()
                padded_line = SENTENCE_START+" "+line_without_fullstop+" "+SENTENCE_END
                padded_line = padded_line.lower()
                # print(padded_line)
                self.count_unigrams(padded_line)
                self.count_bigrams(padded_line)

    def count_unigrams(self, sentence):
        separate_words = sentence.split()
        for word in separate_words:
            if self.unigram_frequencies.get(word, 0) == 0: self.unique_words +=1
            self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
            if word != SENTENCE_START and word != SENTENCE_END:
                self.corpus_length += 1
        self.training_documents.append(separate_words)

    def count_bigrams(self, sentence):
        separate_words = self.get_bigram(sentence)
        # print(separate_words)
        for word1, word2 in separate_words:
            self.bigram_frequencies[(word1, word2)] = self.bigram_frequencies.get((word1, word2), 0) + 1
            self.bigram_corpus_length += 1

        # self.training_documents.append(separate_words)

    def get_bigram(self, sentence):
        result = []
        arr = sentence.split()
        for i in range(len(arr) - 1):
            result.append((arr[i], arr[i + 1]))
        return result

    def prepare_questions(self):
        sentences = list()
        with open(self.questions_file) as f:
            for line in f:
                separate_words = line.lower().split()
                possible_words = separate_words[-1]
                separate_words = separate_words[:-2] # remove the ":" from the question
                separate_words.append(SENTENCE_END)
                separate_words.insert(0, SENTENCE_START)
                possible_words = possible_words.split("/")
                sentences.append((separate_words, possible_words))
        self.sentences = sentences
        return sentences

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
                print("correct: ", words[0], ": ", using_word0)
                print("false: ", words[1], ": ", using_word1)
            else:
                print("correct: ", words[1], ": ", using_word1)
                print("false: ", words[0], ": ", using_word0)

    def process_sentence(self, sentence, smoothing):
        probability = 1
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

    def test_correct(self):
        sum = 0
        for word in self.unigram_frequencies:
            if word != SENTENCE_START and word != SENTENCE_END:
                sum += self.unigram_frequencies[word]
        print(sum / self.corpus_length)

        sum = 0
        for word in self.bigram_frequencies:
            if word != SENTENCE_START and word != SENTENCE_END:
                sum += self.bigram_frequencies[word]
        print(sum / self.bigram_corpus_length)

unigram_nlp = unigram_nlp()
unigram_nlp.process_training_data()
print("------ BASIC UNIGRAM NLM ------")
# print(len(unigram_nlp.training_documents))
print("Unique words:", unigram_nlp.unique_words)
print("Coprus lenght:", unigram_nlp.corpus_length)
print(unigram_nlp.test_correct())
unigram_nlp.prepare_questions()
unigram_nlp.calculate_questions()
# print(unigram_nlp.unigram_frequencies)


print("------ BASIC BIGRAM NLM ------")
bigram_nlp = bigram_nlp(False)
bigram_nlp.process_training_data()
print("Unique words:", bigram_nlp.unique_words)
print("Corpus length:", bigram_nlp.corpus_length)
print("Bigram Corpus lenght:", bigram_nlp.bigram_corpus_length)
print(bigram_nlp.test_correct())
bigram_nlp.prepare_questions()
bigram_nlp.calculate_questions(0)
print("------ SMOOTHING BIGRAM NLM ------")
bigram_nlp.prepare_questions()
bigram_nlp.calculate_questions(.5)

print(bigram_nlp.training_documents[0])
print(bigram_nlp.unigram_frequencies["a"])
print(bigram_nlp.bigram_frequencies.get(("<s>", "a"), 0))
print(bigram_nlp.bigram_frequencies.get(("minga", "chunga"), 0))