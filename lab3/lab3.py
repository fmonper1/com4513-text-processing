import argparse
import itertools
import operator
import random
import sys
from sklearn.metrics import f1_score

########################################
#                Setup                 #
########################################

# 0 = Phi1, 1 = Phi1+Phi2, 2 = Phi1+Phi2+Phi3+Phi4
execution_mode_1 = True
execution_mode_2 = False
execution_mode_3 = False
remove_small_count_keys = False
NUM_ITERATIONS = 11
BEGINING_TOKEN = "<s>"
POSSIBLE_LABELS = ["O", "PER", "LOC", "ORG", "MISC"]


########################################
#              Load data               #
########################################

def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets=[]
    inputs=[]
    zip_inps=[]
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words=[token_vocab[w.strip()]if to_idx else w.strip()for w in sent.split()]
            ner_tags=[target_vocab[w.strip()]if to_idx else w.strip()for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else(inputs, targets)


#######################################
# Current word-current label,Φ1(x, y) #
#######################################
def cw_cl_counts(train_data):
    counts = dict()
    for sentence in train_data:
        for word, label in sentence:
            key = word+"_"+label
            counts[key] = counts.get(key, 0) + 1
            # print(word, label)
    # print(counts)
    return counts


def phi_1(words, labels,  cw_cl_counts):
    cw_cl_in_sentence = dict()

    for i in range(len(words)):
        key = words[i]+"_" +labels[i]
        in_corpus = cw_cl_counts.get(key, 0)
        cw_cl_in_sentence[key] = cw_cl_in_sentence.get(key, 0) + (1 if in_corpus > 0 else 0)
    return cw_cl_in_sentence
    # HERE WHAT DO WE REALLY HAVE TO RETURN, THE COUNTS FROM ¿cw_cl_counts? OR NEW COUNTS


########################################
# Previous label-current labelΦ2(x, y) #
########################################
def pl_cl_counts(train_data):
    counts = dict()
    for sentence in train_data:
        words, labels = list(zip(*sentence))
        labels = (BEGINING_TOKEN, ) + labels
        # print(words, labels)

        bigram_counts = list(find_ngrams(labels, 2))
        bigram_counts = format_labels(bigram_counts)
        # print(bigram_counts)
        for key in bigram_counts:
            counts[key] = counts.get(key, 0) + 1
    # print(counts)
    return counts


def phi_2(words, labels, pl_cl_counts):
    pl_cl_in_sentence = dict()
    labels = (BEGINING_TOKEN ,) + labels
    #print(words, labels)

    bigram_counts = list(find_ngrams(labels, 2))
    bigram_counts = format_labels(bigram_counts)
    # print(bigram_counts)
    for key in bigram_counts:
        in_corpus = pl_cl_counts.get(key, 0)
        pl_cl_in_sentence[key] =  pl_cl_in_sentence.get(key, 0)+ (1 if in_corpus > 0 else 0)

    # print(pl_cl_in_sentence)
    return pl_cl_in_sentence

#########################################
# Current suffix-current label,Φ3(x, y) #
#########################################
def csuff_cl_counts(train_data):
    counts = dict()
    for sentence in train_data:
        for word, label in sentence:
            key = label+"_"+word[-3:]
            counts[key] = counts.get(key, 0) + 1
            # print(word, label)
    # print(counts)
    return counts

def phi_3(words, labels,  csuff_cl_counts):
    csuff_cl_in_sentence = dict()

    for i in range(len(words)):
        key = labels[i]+"_" +words[i][-3:]
        in_corpus = csuff_cl_counts.get(key, 0)
        csuff_cl_in_sentence[key] = csuff_cl_in_sentence.get(key, 0) + (1 if in_corpus > 0 else 0)
    return csuff_cl_in_sentence
    # HERE WHAT DO WE REALLY HAVE TO RETURN, THE COUNTS FROM ¿cw_cl_counts? OR NEW COUNTS


##########################################
# Previous label-current suffix Φ4(x, y) #
##########################################
def pl_csuff_counts(train_data):
    counts = dict()
    for sentence in train_data:
        words, labels = list(zip(*sentence))
        labels = (BEGINING_TOKEN, ) + labels
        # print(words, labels)
        for i in range(len(words)):
            key = labels[i] + "_" + words[i][-3:]
            counts[key] = counts.get(key, 0) + 1
    return counts


def phi_4(words, labels, pl_csuff_counts):
    pl_csuff_in_sentence = dict()
    labels = (BEGINING_TOKEN ,) + labels


    for i in range(len(words)):
        key = labels[i] + "_" + words[i][-3:]
        in_corpus = pl_csuff_counts.get(key, 0)
        pl_csuff_in_sentence[key] = pl_csuff_in_sentence.get(key, 0) + (1 if in_corpus > 0 else 0)
    return pl_csuff_in_sentence


########################################
#              Utilities               #
########################################
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def format_labels(list_of_labels):
    formatted_labels = list()
    for entry in list_of_labels:
        converted_label = ""
        for i in range(len(entry)):
            if i == 0:
                converted_label = entry[i]
            else:
                converted_label = converted_label+"_"+entry[i]
        formatted_labels.append(converted_label)
    return formatted_labels


########################################
#              Training               #
########################################

def train_weights(training_data, phi1_counts, phi2_counts, phi3_counts, phi4_counts):
    weights = dict()
    random.seed(420)
    for i in range(NUM_ITERATIONS):
        print("iteration ", i+1, " of ", NUM_ITERATIONS)
        random.shuffle(training_data)
        for sentence in training_data:
            words, labels = list(zip(*sentence))
            # print("## training new sentence ##")
            # print("len words", len(words))
            # print(words)
            # print(labels)
            possible_label_combinations = list(itertools.product(POSSIBLE_LABELS, repeat=len(words)))

            # pass the iteration number to the prediction to average the weight when doing the prediction
            # we pass i+1 becaus i is 0 in the first iteration
            predicted_y, predicted_counter = do_prediction(words, possible_label_combinations, weights, i+1, phi1_counts, phi2_counts, phi3_counts, phi4_counts)
            actual_y = labels
            if execution_mode_1:
                actual_counter = phi_1(words, labels, phi1_counts)
            if execution_mode_2:
                actual_counter = {**phi_1(words, labels, phi1_counts), **phi_2(words, labels, phi2_counts)}
            if execution_mode_3:
                actual_counter = {**phi_1(words, labels, phi1_counts), **phi_2(words, labels, phi2_counts),
                          **phi_3(words, labels, phi3_counts), **phi_4(words, labels, phi4_counts)}

            # here we combine the PHIs
            if predicted_y != actual_y:
                for key, value in predicted_counter.items():
                    weights[key] = weights.get(key,0) - value
                for key, value in actual_counter.items():
                    weights[key] = weights.get(key,0) + value

    # Average the weights
    for key, value in weights.items():
        weights[key] = weights.get(key, 0) / NUM_ITERATIONS
    return weights


def do_prediction(words, possible_labels, weights, iter_number, phi1_counts, phi2_counts, phi3_counts, phi4_counts):
    # here we combine the PHIs
    max_count = 0
    max_labels = possible_labels[0] #initialize with a value
    label_counts = dict()
    for labels in possible_labels:
        count = 0
        if execution_mode_1:
            result = phi_1(words, labels, phi1_counts)
        if execution_mode_2:
            result = {**phi_1(words, labels, phi1_counts), **phi_2(words, labels, phi2_counts)}
        if execution_mode_3:
            result = {**phi_1(words, labels, phi1_counts), **phi_2(words, labels, phi2_counts),
                      **phi_3(words, labels, phi3_counts), **phi_4(words, labels, phi4_counts)}

        for key, value in result.items():
            count += ((weights.get(key,0) / iter_number) * value)

        if count > max_count:
            max_count = count
            max_labels = labels
            label_counts = result

    return max_labels, label_counts


########################################
#               Testing                #
########################################

def analyze_test_data(test_data, weights, phi1_counts, phi2_counts, phi3_counts, phi4_counts):
    f1_predicted_y = list()
    f1_actual_y = list()
    for sentence in test_data:
        words, labels = list(zip(*sentence))
        possible_label_combinations = list(itertools.product(POSSIBLE_LABELS, repeat=len(words)))

        predicted_y, predicted_counter = do_prediction(words, possible_label_combinations, weights, 1, phi1_counts, phi2_counts, phi3_counts, phi4_counts)
        actual_y = labels
        # actual_counter = {**phi_1(words, labels, phi1_counts), **phi_2(words, labels, phi2_counts)}

        # print(actual_y)
        # print(predicted_y)

        for item in predicted_y:
            f1_predicted_y.append(item)
        for item in actual_y:
            f1_actual_y.append(item)

    f1_micro = f1_score(f1_actual_y, f1_predicted_y, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    print("f1micro \n", f1_micro)
    return f1_micro

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Structured perceptron imprementation.')
    parser.add_argument('train_file')
    parser.add_argument('test_file')

    args = parser.parse_args()
    train_file = args.train_file
    test_file = args.test_file
    print(args)

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    train_data = load_dataset_sents(train_file)
    test_data = load_dataset_sents(test_file)
    # print(train_data)
    # print(len(train_data))

    ########################################
    #                Setup                 #
    ########################################

    # print("##########################################")
    # print("# Counts for Phi functions               #")
    # print("##########################################")
    # # dicts
    cw_cl_counts = cw_cl_counts(train_data)
    pl_cl_counts = pl_cl_counts(train_data)
    csuff_cl_counts = csuff_cl_counts(train_data)
    pl_csuff_counts = pl_csuff_counts(train_data)
    # # print("phi1 ",cw_cl_counts)
    # # print("phi2 ",pl_cl_counts)
    # # print("phi3 ",csuff_cl_counts)
    # # print("phi4 ",pl_csuff_counts)

    if remove_small_count_keys:
        cw_cl_counts = {k: v for k, v in cw_cl_counts.items() if v >= 4}
        pl_cl_counts = {k: v for k, v in pl_cl_counts.items() if v >= 4}
        csuff_cl_counts = {k: v for k, v in csuff_cl_counts.items() if v >= 4}
        pl_csuff_counts = {k: v for k, v in pl_csuff_counts.items() if v >= 4}


    print("")
    print("")

    print("##########################################")
    print("# Phi 1                                  #")
    print("# -------------------------------------- #")
    print("# Training                               #")
    print("##########################################")
    print("Train data length: ", len(train_data))

    trained_weights = train_weights(train_data,cw_cl_counts, pl_cl_counts, csuff_cl_counts, pl_csuff_counts)

    print("##########################################")
    print("# Testing                                #")
    print("##########################################")
    analyze_test_data(test_data, trained_weights, cw_cl_counts, pl_cl_counts, csuff_cl_counts, pl_csuff_counts)
    print("Total trained weights: ", len(trained_weights.items()))
    #train_weights(train_data[100:125],cw_cl_counts, pl_cl_counts)

    print("##########################################")
    print("# Top ranked features                    #")
    print("##########################################")
    print("weights ", sorted(trained_weights.items(), key=operator.itemgetter(1), reverse=True)[:10])

    print("")
    print("")
    print("##########################################")
    print("# Phi 1 + 2                              #")
    print("# -------------------------------------- #")
    print("# Training                               #")
    print("##########################################")
    execution_mode_1 = False
    execution_mode_2 = True
    print("Train data length: ", len(train_data))

    trained_weights = train_weights(train_data,cw_cl_counts, pl_cl_counts, csuff_cl_counts, pl_csuff_counts)

    print("##########################################")
    print("# Testing                                #")
    print("##########################################")
    analyze_test_data(test_data, trained_weights, cw_cl_counts, pl_cl_counts, csuff_cl_counts, pl_csuff_counts)
    print("Total trained weights: ", len(trained_weights.items()))
    #train_weights(train_data[100:125],cw_cl_counts, pl_cl_counts)

    print("##########################################")
    print("# Top ranked features                    #")
    print("##########################################")
    print("weights ", sorted(trained_weights.items(), key=operator.itemgetter(1), reverse=True)[:10])

    print("")
    print("")
    print("##########################################")
    print("# Phi 1 + 2 + 3 + 4                      #")
    print("# -------------------------------------- #")
    print("# Training                               #")
    print("##########################################")
    execution_mode_2 = False
    execution_mode_3 = True

    print("Train data length: ", len(train_data))

    trained_weights = train_weights(train_data,cw_cl_counts, pl_cl_counts, csuff_cl_counts, pl_csuff_counts)

    print("##########################################")
    print("# Testing                                #")
    print("##########################################")
    analyze_test_data(test_data, trained_weights, cw_cl_counts, pl_cl_counts, csuff_cl_counts, pl_csuff_counts)
    print("Total trained weights: ", len(trained_weights.items()))
    #train_weights(train_data[100:125],cw_cl_counts, pl_cl_counts)

    print("##########################################")
    print("# Top ranked features                    #")
    print("##########################################")
    print("weights ", sorted(trained_weights.items(), key=operator.itemgetter(1), reverse=True)[:10])
