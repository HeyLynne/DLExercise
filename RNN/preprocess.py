#coding=utf-8
import csv
import itertools
import nltk
import numpy

from config import BaseConfig

class PreProcessor(object):
    def __init__(self):
        self._vocabulary_size = BaseConfig.vocabulary_size
        self._unknown_token = BaseConfig.unknown_token
        self._sentence_start_token = BaseConfig.sentence_start_token
        self._sentence_end_token = BaseConfig.sentence_end_token

    def process_word_index(self, fin_path):
        """
        Generate sentence 2 index
        """
        with open(fin_path, "rb") as fin:
            reader = csv.reader(fin,skipinitialspace=True)
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
            sentences = ["%s %s %s" % (self._sentence_start_token, x, self._sentence_end_token) for x in sentences]
            tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
            word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
            vocab = word_freq.most_common(self._vocabulary_size - 1)
            self.index_to_word = [x[0] for x in vocab]
            self.index_to_word.append(self._unknown_token)
            self.word_to_index = dict([(word, i) for i, word in enumerate(self.index_to_word)])
            for i, sent in enumerate(tokenized_sentences):
                tokenized_sentences[i] = [w if w in self.word_to_index else self._unknown_token for w in sent]
            self.X_train = numpy.asarray([[self.word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
            self.Y_train = numpy.asarray([[self.word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    def save_data(self):
        """
        Save data
        """
        with open("train.pkl", "wb") as fout:
            pickle.dump(self.X_train,fout,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.Y_train,fout,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.vocabulary_size,fout,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.index_to_word,fout,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.word_to_index,fout,pickle.HIGHEST_PROTOCOL)
            fout.flush()

# if __name__ == "__main__":
#     file_path = "D:\\python script\\deep_learning\\DL_exercise\\DLExercise\\RNN\\data\\reddit.csv"
#     preprocessor = PreProcessor()
#     preprocessor.process_word_index(file_path)