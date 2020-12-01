import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import sys

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF


class SentenceGetter(object):
    """Class to Get the sentence in this format:
    [(Token_1, Part_of_Speech_1, Tag_1), ..., (Token_n, Part_of_Speech_1, Tag_1)]"""

    def __init__(self, data):
        """Args:
            data is the pandas.DataFrame which contains the above dataset"""
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        """Return one sentence"""
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


class NerModel:
    def __init__(self, dataset_path):
        self.BATCH_SIZE = 512  # Number of examples used in each iteration
        self.EPOCHS = 20  # Number of passes through entire dataset
        self.MAX_LEN = 75  # Max length of review (in words)
        self.EMBEDDING = 40  # Dimension of word embedding vector

        self.model = self.__init_model(dataset_path)

    def __generate_model(self):
        input = Input(shape=(self.MAX_LEN,))
        model = Embedding(input_dim=self.n_words + 2, output_dim=self.EMBEDDING,  # n_words + 2 (PAD & UNK)
                          input_length=self.MAX_LEN, mask_zero=True)(input)  # default: 20-dim embedding
        model = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.1))(model)  # variational biLSTM
        model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        crf = CRF(self.n_tags + 1)  # CRF layer, n_tags+1(PAD)
        out = crf(model)  # output
        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()

        return model

    def __init_model(self, dataset_path):
        print("is_gpu_available: %s" % (tf.test.is_gpu_available(),))

        data = pd.read_csv(dataset_path, encoding="utf-8")
        data = data.fillna(method="ffill")
        print("Number of sentences: ", len(data.groupby(['Sentence #'])))
        words = list(set(data["Word"].values))
        words.sort()
        self.n_words = len(words)
        print("Number of words in the dataset: ", self.n_words)
        tags = list(set(data["Tag"].values))
        tags.sort()
        print("Tags:", tags)
        self.n_tags = len(tags)
        print("Number of Labels: ", self.n_tags)
        print("What the dataset looks like:")
        # Show the first 10 rows
        data.head(n=10)

        getter = SentenceGetter(data)
        sent = getter.get_next()
        print('This is what a sentence looks like:')
        print(sent)

        self.sentences = getter.sentences
        self.word2idx = {w: i + 2 for i, w in enumerate(words)}
        self.word2idx["UNK"] = 1  # Unknown words
        self.word2idx["PAD"] = 0  # Padding
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.tag2idx = {t: i + 1 for i, t in enumerate(tags)}
        self.tag2idx["PAD"] = 0
        self.idx2tag = {i: w for w, i in self.tag2idx.items()}

        X = [[self.word2idx[w[0]] for w in s] for s in self.sentences]
        self.X = pad_sequences(maxlen=self.MAX_LEN, sequences=X, padding="post", value=self.word2idx["PAD"])
        y = [[self.tag2idx[w[2]] for w in s] for s in self.sentences]
        self.y = pad_sequences(maxlen=self.MAX_LEN, sequences=y, padding="post", value=self.tag2idx["PAD"])

        self.model = self.__generate_model()

        return self.model

    def fit(self):
        y = [to_categorical(i, num_classes=self.n_tags + 1) for i in self.y]  # n_tags+1(PAD)
        X_tr, X_te, y_tr, y_te = train_test_split(self.X, y, test_size=0.1)
        X_tr.shape, X_te.shape, np.array(y_tr).shape, np.array(y_te).shape
        print('Raw Sample: ', ' '.join([w[0] for w in self.sentences[0]]))
        print('Raw Label: ', ' '.join([w[2] for w in self.sentences[0]]))
        print('After processing, sample:', self.X[0])
        print('After processing, labels:', self.y[0])

        history = self.model.fit(X_tr, np.array(y_tr), batch_size=self.BATCH_SIZE,
                            epochs=self.EPOCHS, validation_split=0.1, verbose=2)
        return history


    def save(self, model_path):
        self.model.save(model_path)
