from nltk.corpus import movie_reviews
from nltk.stem.porter import PorterStemmer
import random
import nltk
import svmutil
import svm


class SASVM(object):
    
    def init(self):
        print("Initiazing... ")
        self.organize_reviews()
        self.generate_dict()


    def organize_reviews(self):
        #self.reviews = [(list(movie_reviews.words(fileid)), category)
        #               for category in movie_reviews.categories()
        #              for fileid in movie_reviews.fileids(category)]
        #random.shuffle(self.reviews)
        
        reviews_pos = [(list(movie_reviews.words(fileid)), "pos")
                      for fileid in movie_reviews.fileids("pos")]
        reviews_neg = [(list(movie_reviews.words(fileid)), "neg")
                      for fileid in movie_reviews.fileids("neg")]

        self.reviews = []
        cnt1 = cnt2 = 0
        while cnt1 < len(reviews_pos) or cnt2 < len(reviews_neg):
            if cnt1 < len(reviews_pos):
                self.reviews.append(reviews_pos[cnt1])
                cnt1 += 1
            if cnt2 < len(reviews_neg):
                self.reviews.append(reviews_neg[cnt2])
                cnt2 += 1


    def remove_symbols(self, words):
        non_symbols = []
        for word in words:
            if str.isalnum(str(word.lower())) is True:
                non_symbols.append(word.lower())
        
        return non_symbols


    def remove_stop_words(self, words):
        stop_words = set(nltk.corpus.stopwords.words("english"))
        non_stop_words = []
        for word in words:
            if word.lower() not in stop_words:
                non_stop_words.append(word.lower())

        return non_stop_words


    def do_stemming(self, words):
        stemmer = PorterStemmer()
        stemmed_words = []
        for word in words:
            stemmed_words.append(stemmer.stem(word))

        return stemmed_words


    def generate_dict(self):
       
        words = movie_reviews.words()

        print("Removing Stop Words... ")
        words = self.remove_stop_words(words)
       
        print("Removing Symbols... ")
        words = self.remove_symbols(words)

        print("Stemming using PorterStemmer... ")
        words = self.do_stemming(words)

        print("Generating Vocabulary... ")
        word_freqdist = nltk.FreqDist(w.lower() for w in words)

        N = 4000
        print("Keeping Only %s Most Frequent Words... " % N) 
        self.word_dict = [wd for (wd, cnt) in word_freqdist.most_common(N)]

        print("Generating MFW.vocab File of Vocab... ")
        f = open("MFW.vocab", "w")
        for word in self.word_dict:
            f.write("%s\n" % word)
        f.close()


    def review_features(self, review):
        words = set(review)
        words = set(self.do_stemming(words))
        features = {}
        for word in self.word_dict:
            features['%s' % word] = (word in words)

        return features

 
    def NaiveBayes(self):
        print("\n====================== Using NaiveBayes Classifier =====================\n")
        print("Generating FeatureSets... ")
        featuresets = [(self.review_features(r), c) for (r,c) in self.reviews]
        train_set, test_set = featuresets[100:], featuresets[:100]
        
        print("Training... on (%s) samples" % len(train_set))
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        
        print("Testing...  on (%s) samples" % len(test_set))
        print("Accuracy : %s" % nltk.classify.accuracy(classifier, test_set))
        #classifier.show_most_informative_features(5)


    def GenerateFeatureFiles(self):
        print("Generating FeatureSets... ")
        featuresets = [(self.review_features(r), c) for (r,c) in self.reviews]
        train_set, test_set = featuresets[100:], featuresets[:100]

        print("Generating Featurefile(BOW_train.feat)... ")
        f = open("BOW_train.feat", "w")
        
        for fvector, label in train_set:
            if label == "neg":
                f.write("-1")
            else:
                f.write("1")
            
            cnt = 1
            for key, value in fvector.items():
                if value == True:
                    f.write(" %s:1" % cnt)
                cnt += 1
            f.write("\n")
        f.close()

        print("Generating Featurefile(BOW_test.feat)... ")
        f = open("BOW_test.feat", "w")
        
        for fvector, label in test_set:
            if label == "neg":
                f.write("-1")
            else:
                f.write("1")
            
            cnt = 1
            for key, value in fvector.items():
                if value == True:
                    f.write(" %s:1" % cnt)
                cnt += 1
            f.write("\n")
        f.close()


    def SVM(self, useFeatureFiles = True):
        print("\n====================== Using Support Vector Machines =====================\n")

        if useFeatureFiles is not True:
            self.GenerateFeatureFiles()

        print("Reading Feature Files for Training... ")
        y, x = svmutil.svm_read_problem("BOW_train.feat")

        print("Training... on %s Samples" % len(y))
        m = svmutil.svm_train(y, x, '-c 0.5 -g 0.002 -r 78.8889 -t 1')
       
        print("Reading Feature Files for Testing... ")
        y, x = svmutil.svm_read_problem("BOW_test.feat")
        
        print("Testing... on %s Samples" % len(y))
        p_label, p_acc, p_val = svmutil.svm_predict(y, x, m)

    def start(self, useFeatureFiles = True):
        if useFeatureFiles is not True:
            self.init()

        #self.NaiveBayes()
        self.SVM(useFeatureFiles)
        
if __name__ == '__main__':
    SASVM().start(True)
