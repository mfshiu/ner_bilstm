import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF


class SentenceGetter(object):
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


BATCH_SIZE = 512  # Number of examples used in each iteration
EPOCHS = 20  # Number of passes through entire dataset
MAX_LEN = 75  # Max length of review (in words)
EMBEDDING = 40  # Dimension of word embedding vector
LSTM_UNITS = 200

class NerModel:
    def __init__(self, dataset_path, embedding_size = EMBEDDING, max_words = MAX_LEN):
        self.MAX_LEN = max_words  # Max length of review (in words)
        self.EMBEDDING = embedding_size  # Dimension of word embedding vector

        self.WORD_PAD_INDEX = 0
        self.WORD_UNK_INDEX = 1
        self.TAG_PAD_INDEX = 0

        self.model = self.__init_model(dataset_path)

    def __generate_model(self):
        input = Input(shape=(self.MAX_LEN,))
        model = Embedding(input_dim=self.n_words + 2, output_dim=self.EMBEDDING,  # n_words + 2 (PAD & UNK)
                          input_length=self.MAX_LEN, mask_zero=True)(input)  # default: 20-dim embedding
        model = Bidirectional(LSTM(units=LSTM_UNITS, return_sequences=True,
                                   recurrent_dropout=0.2))(model)  # variational biLSTM
        model = TimeDistributed(Dense(LSTM_UNITS, activation="relu"))(model)  # a dense layer as suggested by neuralNer
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
        self.word2idx["UNK"] = self.WORD_UNK_INDEX  # Unknown words
        self.word2idx["PAD"] = self.WORD_PAD_INDEX  # Padding
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.tag2idx = {t: i + 1 for i, t in enumerate(tags)}
        self.tag2idx["PAD"] = self.TAG_PAD_INDEX
        self.idx2tag = {i: w for w, i in self.tag2idx.items()}

        X = [[self.word2idx[w[0]] for w in s] for s in self.sentences]
        self.X = pad_sequences(maxlen=self.MAX_LEN, sequences=X, padding="post", value=self.word2idx["PAD"])
        y = [[self.tag2idx[w[2]] for w in s] for s in self.sentences]
        self.y = pad_sequences(maxlen=self.MAX_LEN, sequences=y, padding="post", value=self.tag2idx["PAD"])

        self.model = self.__generate_model()

        return self.model

    def fit(self, batch_size=BATCH_SIZE, epochs=EPOCHS):
        print("Fitting, batch_size: %d, epochs: %d" % (batch_size, epochs), end="")
        y = [to_categorical(i, num_classes=self.n_tags + 1) for i in self.y]  # n_tags+1(PAD)
        X_tr, X_te, y_tr, y_te = train_test_split(self.X, y, test_size=0.1)
        X_tr.shape, X_te.shape, np.array(y_tr).shape, np.array(y_te).shape
        print('Raw Sample: ', ' '.join([w[0] for w in self.sentences[0]]))
        print('Raw Label: ', ' '.join([w[2] for w in self.sentences[0]]))
        print('After processing, sample:', self.X[0])
        print('After processing, labels:', self.y[0])

        history = self.model.fit(X_tr, np.array(y_tr), batch_size,
                            epochs, validation_split=0.1, verbose=2)
        print("done.")
        return history


    def get_word_index(self,word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.WORD_PAD_INDEX


    def load_weights(self, model_path):
        print("Load from: %s ..."%(model_path,), end="")
        self.model.load_weights(model_path)
        print("done.")


    # sentences = [
    #   ['word', 'word', ... ,'word'],
    #   ['word', 'word', ... ,'word'],
    #   ...
    #   ['word', 'word', ... ,'word'],
    # ]
    def predict(self, sentences):
        X = [[self.get_word_index(w) for w in s] for s in sentences]
        X = pad_sequences(maxlen=self.MAX_LEN, sequences=X, padding="post", value=self.WORD_PAD_INDEX)
        pred_cat = self.model.predict(X)
        pred = np.argmax(pred_cat, axis=-1)

        return pred

    def save(self, model_path):
        print("Save to: %s ..."%(model_path,), end="")
        self.model.save(model_path)
        print("done.")
