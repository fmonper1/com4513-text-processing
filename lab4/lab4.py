###imports 

from collections import Counter
import sys
import itertools
import numpy as np
import time, random
from sklearn.metrics import f1_score
from collections import OrderedDict
import matplotlib.pyplot as plt

random.seed(11242)

depochs = 5
feat_red = 0

DO_VITERBI = False
DO_BEAM = False
beam_size = 3

print("\nDefault no. of epochs: ", depochs)
print("\nDefault feature reduction threshold: ", feat_red)


print("\nLoading the data \n")


"""Loading the data"""

### Load the dataset
def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

# print(sys.argv[1])
if sys.argv[1].lower() == "-v":
    DO_VITERBI = True

if sys.argv[1].lower() == "-b":
    DO_BEAM = True

train_data = load_dataset_sents(sys.argv[2])
test_data = load_dataset_sents(sys.argv[3])

## unique tags
all_tags = ["O", "PER", "LOC", "ORG", "MISC"]

""" Defining our feature space """

print("\nDefining the feature space \n")


# feature space of cw_ct
def cw_ct_counts(data, freq_thresh = 5): # data inputted as (cur_word, cur_tag)
   
    cw_c1_c = Counter()
   
    for doc in data:

        cw_c1_c.update(Counter(doc))
    
    
    
    return Counter({k:v for k,v in cw_c1_c.items() if v > freq_thresh})

cw_ct_count = cw_ct_counts(train_data, freq_thresh = feat_red)

# feature representation of a sentence cw-ct
def phi_1(sent, cw_ct_counts): # sent as (cur_word, cur_tag)
    
    phi_1 = Counter()
    
    # include features only if found in feature space
    phi_1.update([item for item in sent if item in cw_ct_count.keys()])
    
    return phi_1

sent = train_data[0]

phi_1(sent, cw_ct_count)

# feature space of pt-ct
def pt_ct_counts(data, freq_thresh = 5): # input (cur_word, cur_tag) 
    
    tagtag = Counter()
    
    for doc in data:
    
        tags = list(zip(*doc))[1]

        for i in range(len(tags)):

            if i == 0:

                tagtag.update([("*", tags[i])])

            else:

                tagtag.update([(tags[i-1], tags[i])])

    # return feature space with features with counts above freq_thresh
    return Counter({k:v for k,v in tagtag.items() if v > freq_thresh})

pt_ct_count = pt_ct_counts(train_data, freq_thresh = feat_red)

# combining feature spaces
comb_featspaces = pt_ct_count + cw_ct_count

# creating our sentence features
def phi_2(sent, pt_ct_count):
    
    sentence, tags = zip(*sent)
    
    tags = ["*"] + list(tags)
    
    # returning features if found in the feature space
    tags = [(tags[i], tags[i+1]) for i in range(len(tags)-1) if (tags[i], tags[i+1]) in pt_ct_count]
    
    return Counter(tags)
    
sent = train_data[0]    
phi_2(sent, pt_ct_count)

"""Perceprton"""

class Perceptron():
    
    def __init__(self,all_tags):
        super(Perceptron, self).__init__()
        self.all_tags = all_tags
        
    # creating all possible combinaions of 
    def pos_combos(self,sentence):
        
        combos = [list(zip(sentence, p)) for p in itertools.product(self.all_tags,repeat=len(sentence))]
        
        return combos

    def do_viterbi(self,doc, weights, extra_feat = True):
        # unzippin them
        sentence, tags = list(zip(*doc))

        #backpointer
        bckpntr = dict()
        V = OrderedDict()

        #initialize values for <s> as 0
        for tag in all_tags:
            V[(tag, 0)] = 0

        # add <s> to begining of sentence
        sentence = ("<s>",) + sentence

        for n in range(1, len(sentence)):
            scores = np.zeros(len(all_tags))
            for y in all_tags:
                # initialize value in V dictionary

                sent_tag = [(sentence[n], y)]
                phi = phi_1(sent_tag, cw_ct_count)
                phi_times_weight = 0
                for pair in phi:
                    if pair in weights:
                        phi_times_weight += weights[pair] * phi[pair]
                    else:
                        phi_times_weight += 0

                    # store the score with the index
                scores = np.zeros(len(all_tags))

                # use this to loop over all previous values
                for i in range(len(all_tags)):
                    prev_tag = all_tags[i]
                    scores[i] = V[(prev_tag, n-1)] + phi_times_weight

                max_index = np.argmax(scores)
                max_score = np.max(scores)
                V[(y, n)] = max_score
                bckpntr[(y, n)] = ( all_tags[max_index], n-1)

        return_tags = self.process_backpointer(sentence, V, bckpntr)
        # remove the start tokens
        to_return = list(zip(sentence[1:], return_tags[1:]))
        return to_return

    def process_backpointer(self, sentence, V, bckpntr):
        start = True
        tag, idx = max(V, key=V.get)
        predicted_tags = (tag,)
        c = len(sentence)
        last_result = ()
        while c > 0:
            if start:
                dict_key = max(V)
                start = False
            else:
                dict_key = last_result

            result = bckpntr[dict_key]
            tag, idx = result
            c = idx
            predicted_tags = (tag,) + predicted_tags
            last_result = result

        return predicted_tags

    def do_beam_search(self, doc, weights):
        sentence, tags = list(zip(*doc))
        beam = dict()
        beam[('<s>',)] = 0

        for n in range(len(sentence)):
            new_beam = {}
            for b in beam.keys():
                for y in all_tags:
                    sent_tag = list(zip(sentence, b[1:] + (y,)))
                    phi = phi_1(sent_tag, cw_ct_count)
                    score = 0
                    for pair in phi:
                        if pair in weights:
                            score += weights[pair] * phi[pair] + beam[b]
                    new_beam[b + (y,)] = score
            beam = dict(Counter(new_beam).most_common(beam_size))

        best_in_beam = dict(Counter(beam).most_common(1))
        for item in best_in_beam.keys():
            best_in_beam = item

        return list(zip(sentence, best_in_beam[1:]))


    def scoring(self, doc, weights, extra_feat=True):

        # unzippin them
        sentence, tags = list(zip(*doc))

        # all possible combos of sequences
        combos = list(enumerate(self.pos_combos(sentence)))
        # our score matrix
        scores = np.zeros(len(combos))

        # looping through all possible combos
        for index, sent_tag in combos:

            if extra_feat is False:

                # retrieving the counter if its in our feature space
                phi = phi_1(sent_tag, cw_ct_count)

            else:

                phi1 = phi_1(sent_tag, cw_ct_count)
                phi2 = phi_2(sent_tag, pt_ct_count)

                phi = phi1 + phi2

            # if its not then the score is 0
            if len(phi) == 0:

                scores[index] = 0

            else:

                temp_score = 0

                # otherwise do the w*local_phi
                for pair in phi:

                    if pair in weights:

                        temp_score += weights[pair] * phi[pair]

                    else:

                        temp_score += 0

                # store the score with the index
                scores[index] = temp_score

        # retrieve the index of the highest scoring sequence
        max_scoring_position = np.argmax(scores)

        # retrieve the highest scoring sequence
        max_scoring_seq = combos[max_scoring_position][1]

        # print(max_scoring_seq)
        return max_scoring_seq
    
    def train_perceptron(self, data, epochs, shuffle = True, extra_feat = False):
    
        # variables used as metrics for performance and accuracy
        iterations = range(len(data)*epochs)
        false_prediction = 0
        false_predictions = []
        
        # initialising our weights dictionary as a counter
        # counter.update allows addition of relevant values for keys
        # a normal dictionary replaces the key-value pair
        weights = Counter()
    
        start = time.time()
    
        # multiple passes
        for epoch in range(epochs):
            false = 0
            now = time.time()
            
            # going through each sentence-tag_seq pair in training_data
            
            # shuffling if necessary
            if shuffle == True:
            
                random.shuffle(data)
            
            for doc in data:
                
                # retrieve the highest scoring sequence
                if DO_VITERBI:
                    max_scoring_seq = self.do_viterbi(doc, weights, extra_feat=extra_feat)
                elif DO_BEAM:
                    max_scoring_seq = self.do_beam_search(doc, weights)
                else:
                    max_scoring_seq = self.scoring(doc, weights, extra_feat = extra_feat)

                # print(max_scoring_seq)
                
                # if the prediction is wrong
                if max_scoring_seq != doc:

                    correct = Counter(doc)
                    
                    # negate the sign of predicted wrong
                    predicted = Counter({k:-v for k,v in Counter(max_scoring_seq).items()})
    
                    # add correct
                    weights.update(correct)
                    
                    # negate false
                    weights.update(predicted)
    
                    
                    """Recording false predictions"""
                    false += 1
                    false_prediction += 1
                false_predictions.append(false_prediction)
    
                
            print("Epoch: ", epoch+1, 
                  " / Time for epoch: ", round(time.time() - now,2),
                 " / No. of false predictions: ", false)

            
        return weights, false_predictions, iterations
    
    # testing the learned weights
    def test_perceptron(self,data, weights, extra_feat = False):
        
        correct_tags = []
        predicted_tags = []
        
        i = 0
        
        for doc in data:
    
            _, tags = list(zip(*doc))
            
            correct_tags.extend(tags)

            if DO_VITERBI:
                max_scoring_seq = self.do_viterbi(doc, weights, extra_feat=extra_feat)
            elif DO_BEAM:
                max_scoring_seq = self.do_beam_search(doc, weights)
            else:
                max_scoring_seq = self.scoring(doc, weights, extra_feat=extra_feat)

            _, pred_tags = list(zip(*max_scoring_seq))
            
            predicted_tags.extend(pred_tags)
    
        return correct_tags, predicted_tags
    
    def evaluate(self, correct_tags, predicted_tags):
        
        f1 = f1_score(correct_tags, predicted_tags, average='micro', labels=self.all_tags)
        
        print("F1 Score: ", round(f1, 5))
        
        return f1


if DO_VITERBI:
    print("Using VITERBI")
elif DO_BEAM:
    print("Using BEAM")
else:
    print("Using PERCEPTRON")

perceptron = Perceptron(all_tags)

print("\nTraining the perceptron with (cur_word, cur_tag) \n")

weights, false_predictions, iterations = perceptron.train_perceptron(train_data, epochs = depochs)

print("\nEvaluating the perceptron with (cur_word, cur_tag) \n")

correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights)

f1 = perceptron.evaluate(correct_tags, predicted_tags)

# print("\nTraining the perceptron with (cur_word, cur_tag) & (prev_tag, current_tag) \n")
#
# weights, false_predictions, iterations = perceptron.train_perceptron(train_data, epochs = depochs, extra_feat=True)
#
# print("\nTraining the perceptron with (cur_word, cur_tag) & (prev_tag, current_tag) \n")
#
# correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights, extra_feat=True)
#
# f1 = perceptron.evaluate(correct_tags, predicted_tags)
#
#