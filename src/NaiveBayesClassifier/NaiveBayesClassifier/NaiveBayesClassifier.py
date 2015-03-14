import math
from nltk.probability import FreqDist
from _collections import defaultdict

class NaiveBayesClassifier(object):

    def __init__(self, vocab, label_feature_freqdist, label_probdist):
        self._vocab = vocab
        self.label_feature_freqdist = label_feature_freqdist
        self._label_probdist = label_probdist

    @staticmethod
    def train(labeled_featuresets):
        
        label_doc_count = defaultdict(float)
        vocab = defaultdict(float)
        label_feature_freqdist = defaultdict(lambda: defaultdict(int))

        for feature_set, label in labeled_featuresets:
            label_doc_count[label] += 1
            for fname, fval in feature_set.items():
                    vocab[fname] = 1
                    label_feature_freqdist[label][fname] += 1

        total_documents = float(sum(label_doc_count.values()))
        label_probdist = defaultdict(float)

        """for feature_set, label in labeled_featuresets:
            print("label = ", label)
            for fname in feature_set.items():
                print( label_feature_freqdist[label][fname])"""

        for label, freq in label_doc_count.items():
            label_probdist[label] = freq / total_documents
                
        return NaiveBayesClassifier(vocab, label_feature_freqdist, label_probdist)

    def prob_classify(self, featureset):
        label_scores = {}
        vocab_cnt = float(sum(self._vocab.values()))

        for label in self._label_probdist:
            score = math.log(self._label_probdist[label])

            label_fname_cnt = float(sum(self.label_feature_freqdist[label].values()))

            for fname, fval in featureset.items():
                fname_cnt = self.label_feature_freqdist[label][fname]
                score += math.log((fname_cnt + 1.0) / (label_fname_cnt + vocab_cnt))

            label_scores[label] = score

        return label_scores

    def classify(self, featureset):
        label_scores = self.prob_classify(featureset)
        sorted_scores = sorted(label_scores.items(), key = lambda x: x[1], reverse = True)
        return sorted_scores[0][0]