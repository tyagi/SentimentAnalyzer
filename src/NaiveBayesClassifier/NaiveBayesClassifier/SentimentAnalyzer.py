import nltk
from NaiveBayesClassifier import NaiveBayesClassifier
class SentimentAnalyzer(object):
    
    def get_training_data(self, path):
        """
            Retrive all Data in Corpus Root and Convert to PlainTextCorpusReader

        :param path: Path is the path to Corpus Root which contains all files that needs to be extracted
        """
        data = nltk.corpus.PlaintextCorpusReader(path, '.*')
        return data

    def extract_feature(self, document):
        document_word_freq = nltk.FreqDist(word.lower() for word in document)
        features = {}
        for word in document_word_freq:
            features["contains (%s)" % word] = document_word_freq[word]

        return features

    def train(self):
        self._neg_train_data = self.get_training_data(self.data_root + 'train\\neg\\')
        self._pos_train_data = self.get_training_data(self.data_root + 'train\\pos\\')

        print ("Preparing Train Data... ")
        documents = [(list(self._neg_train_data.words(fileid)), "neg")
                     for fileid in self._neg_train_data.fileids()]
        
        documents += [(list(self._pos_train_data.words(fileid)), "pos")
                     for fileid in self._pos_train_data.fileids()]

        print ("Preparing Train FeatureSets... ")
        featuresets = [(self.extract_feature(d), c)  for d, c in documents]
        
        print ("Training... ")
        self.classifier = NaiveBayesClassifier.train(featuresets)

    def test(self):
        self._neg_test_data = self.get_training_data(self.data_root + "test\\neg\\")
        self._pos_test_data = self.get_training_data(self.data_root + "test\\pos\\")
        
        print ("Preparing Test Data... ")
        documents = [(list(self._neg_test_data.words(fileid)), "neg")
                     for fileid in self._neg_test_data.fileids()]
        documents += [(list(self._pos_test_data.words(fileid)), "pos")
                     for fileid in self._pos_test_data.fileids()]
        
        print ("Preparing Test FeatureSets... ")
        featuresets = [(self.extract_feature(d), c)  for d, c in documents]

        match = {}
        print ("Testing... ")
        for document, label in documents:
            if self.classifier.classify(self.extract_feature(document)) == label:
                if label in match.keys():
                    match[label] += 1
                else:
                    match[label] = 1
        
        print ("\nAccuracy :", float(sum(match.values())) / len(documents))
        print (match, "\n")

    def Start(self):
        #self.data_root = 'D:\\Study, Docs and Projects\\Sentimental Analysis\\Reference Material\\'         #Data Outside Project
        self.data_root = ''                                                                                  #Data Within Project

        self.train()
        self.test()


if __name__ == '__main__':
    SentimentAnalyzer().Start()


