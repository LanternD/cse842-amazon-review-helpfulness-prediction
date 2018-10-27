import json
import os
import pickle
import math
import csv
import time
from pathlib import Path
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD, Adadelta
from keras.layers import LSTM, Embedding, SimpleRNN
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from numpy import corrcoef
from sklearn.utils import shuffle
import re
from sklearn.neighbors import KNeighborsClassifier
import csv
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import svm

DATA_PATH = '../proj_data/'
ROOT_PATH = '../'
MAX_LENGTH = 256

class DataPreprocessor(object):
    def __init__(self):
        self.json_path = DATA_PATH + 'json_raw/'
        self.chunk_path = DATA_PATH + 'json_raw/Electronics_5_chunk_'
        self.csv_path = DATA_PATH + 'input_sequence/text_int_seq_'
        self.glove_path = DATA_PATH + 'Glove6B/'
        self.review_text_list = []
        self.review_helpful_score = []
        self.V = None
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []

    def file_chunking(self):
        """
            Divide the large file into smaller ones. 65536 lines per file.
            Require the 'Electronics_5.json' file existing.
        """
        i = 0  # line counter
        j = 0  # chunk counter

        with open(DATA_PATH + 'Electronics_5.json', 'r') as json_raw:
            chunk_file = open(self.chunk_path + str(j) + '.json', 'w')
            while True:
                line_buffer = json_raw.readline()
                i += 1
                if i % 65536 != 0:
                    if not line_buffer:  # end of file
                        break
                    else:
                        chunk_file.write(line_buffer)
                else:
                    chunk_file.close()
                    j += 1
                    chunk_file = open(self.chunk_path + str(j) + '.json', 'w')
            chunk_file.close()
        json_raw.close()

    def load_json_raw_data(self, chunk_num):

        chunk_path = self.chunk_path + str(chunk_num) + '.json'
        with open(chunk_path, 'r') as json_chunk_file:
            for lines in json_chunk_file:
                # line_buf = json_chunk_file.readline()
                line_jsonify = json.loads(lines)  # build a dictionary
                self.review_text_list.append(str(line_jsonify['reviewText']))
                helpful_list = line_jsonify['helpful']
                # if hlpfl_list[1] == 0:
                #     hlpfl_score = 0.6
                # else:
                #     hlpfl_score = hlpfl_list[0] / hlpfl_list[1]  # smoothing
                if helpful_list[1] == 0 or helpful_list[0] == helpful_list[1] / 2:
                    helpful_score = 0  # Neutral
                elif helpful_list[0] > helpful_list[1] / 2:
                    helpful_score = 1  # helpful
                elif helpful_list[0] < helpful_list[1] / 2:
                    helpful_score = 2  # not helpful
                else:
                    helpful_score = 3  # abnormal cases
                self.review_helpful_score.append(helpful_score)
            # print(self.review_helpful_score)
            json_chunk_file.close()
        # print(self.review_text_list[6])
        # print(self.review_text_list[7])
        print('Chunk ' + str(chunk_num) + ' loaded. Label count: ')
        count = [0, 0, 0, 0]
        for item in self.review_helpful_score:
            if item == 0:
                count[0] += 1
            elif item == 1:
                count[1] += 1
            elif item == 2:
                count[2] += 1
            else:
                count[3] += 1
        print(count)

    def vocabulary_building(self):
        vocab_pkl = Path(ROOT_PATH + 'Review_Vocabulary.pkl')
        if vocab_pkl.is_file():
            with open('Review_Vocabulary.pkl', 'rb') as f_pkl:
                self.V = pickle.load(f_pkl)
                f_pkl.close()
        else:
            self.V = Vocabulary()  # a vocabulary object
            with open('Review_Vocabulary.pkl', 'wb') as f_pkl:
                pickle.dump(self.V, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
                f_pkl.close()

        for review_text in self.review_text_list:
            # print(self.review_text_list[0][0])
            xx = re.findall(r"[\w']+|[.,!?;]", review_text)  # use regular expression to split the sequence
            for word in xx:
                self.V.add_word(word)
        print(self.V.word_count)
        # print(self.vocabulary.vocabulary)
        with open('Review_Vocabulary.pkl', 'wb') as f_pkl:
            pickle.dump(self.V, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
            f_pkl.close()

    def matrixing_sequences(self, chunk_num):
        # Process only 1 chunk each time this function is called.
        vocab_pkl = Path(ROOT_PATH + 'Review_Vocabulary.pkl')
        if vocab_pkl.is_file():
            with open('Review_Vocabulary.pkl', 'rb') as f_pkl:
                self.V = pickle.load(f_pkl)
                f_pkl.close()
        else:
            print('Vocabulary pickling file does not exist. Quit.')

        with open(self.csv_path + str(chunk_num) + '.csv', 'w', newline='') as seq_csv_file:
            csv_writer = csv.writer(seq_csv_file, quoting=csv.QUOTE_MINIMAL)
            for i in range(len(self.review_text_list)):
                single_line = [self.review_helpful_score[i]]
                word_index_seq = []
                xx = re.findall(r"[\w']+|[.,!?;]", self.review_text_list[i])
                # print(xx)
                for word in xx:
                    word_index_seq.append(self.V.word_indexing(word))
                single_line += word_index_seq
                # print(single_line)
                csv_writer.writerow(single_line)
            seq_csv_file.close()

    def train_test_loading(self, training_chunk_num_list, test_chunk_num):
        for chk_num in training_chunk_num_list:
            with open(self.csv_path + str(chk_num) + '.csv', 'r', newline='') as csv_raw_f:
                csv_reader = csv.reader(csv_raw_f)
                total_list = list(csv_reader)
                csv_raw_f.close()
            for lines in total_list:
                self.X_train.append(lines[1:])
                self.Y_train.append(lines[0])  # truth label

        with open(self.csv_path + str(test_chunk_num) + '.csv', 'r', newline='') as csv_raw_f:
            csv_reader = csv.reader(csv_raw_f)
            total_list = list(csv_reader)
            csv_raw_f.close()
            for lines in total_list:
                self.X_test.append(lines[1:])
                self.Y_test.append(lines[0])  # truth label

    def tokenization(self):
        MAX_NB_WORDS = 50000
        tkn = Tokenizer(num_words=MAX_NB_WORDS, lower=True, split=' ')
        tkn.fit_on_texts(self.review_text_list)
        print(tkn.word_index)
        # seqq = tkn.texts_to_sequences(test_seq)
        print('Found %s unique tokens.' % len(tkn.word_index))

        with open(os.path.join(DATA_PATH, 'tokenizer.pkl'), 'wb') as f_pkl:
            pickle.dump(tkn, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
            f_pkl.close()

        # s2mat = tkn.texts_to_matrix(test_seq)
        # s2mat = tkn.sequences_to_matrix(t2seq)
        # print(seqq)
        # print(len(seqq))

    def build_w2v_dict(self):
        embeddings_index = {}
        f = open(os.path.join(self.glove_path, 'glove.6B.100d.txt'), encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Word2Vec dictionary built. Save to local disk')
        with open(os.path.join(self.glove_path, 'w2v_weights.pkl'), 'wb') as f_pkl:
            pickle.dump(embeddings_index, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
            f_pkl.close()

    def load_stop_word(self):
        f = open(os.path.join(DATA_PATH, 'stop_word_list.txt'), encoding='utf-8')
        stop_word_list = f.readline().split(' ')
        f.close()
        print(stop_word_list)
        print(len(stop_word_list))

    def embedding_matrix_build(self):
        w2v_pkl = Path(os.path.join(self.glove_path, 'w2v_weights.pkl'))
        if w2v_pkl.is_file():
            with open(os.path.join(self.glove_path, 'w2v_weights.pkl'), 'rb') as f_pkl:
                w2v_weights = pickle.load(f_pkl)
                f_pkl.close()
        else:
            w2v_weights = None
            print('Run function build_w2v_dict() first.')

        print(len(w2v_weights))
        with open(os.path.join(DATA_PATH, 'tokenizer.pkl'), 'rb') as f_pkl:
            tkn = pickle.load(f_pkl)
            f_pkl.close()
        non_seen_word_count = 0
        non_seen_word = []
        for word in tkn.word_index:
            if word not in w2v_weights:
                non_seen_word.append(word)
                non_seen_word_count += 1
        # print(non_seen_word)
        print('Not seen word:', non_seen_word_count)

    # def word_embedding_weight_matrix(self):
        word_index = tkn.word_index
        embedding_dimension = 100
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
        for word, i in word_index.items():
            # word index starts from 1.
            embedding_vector = w2v_weights.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector[:embedding_dimension]
        print(embedding_matrix.shape)

        with open(os.path.join(DATA_PATH, 'embedding_matrix.pkl'), 'wb') as f_pkl:
            pickle.dump(embedding_matrix, f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
            f_pkl.close()


class DataPreprocessorV2(DataPreprocessor):

    def chunk_json_raw_to_csv(self, chunk_index):

        self.review_text_list = []
        self.review_helpful_score = []
        text_0 = []
        text_1 = []
        text_2 = []
        score_0 = []
        score_1 = []
        score_2 = []
        threshold = 0.6
        num_limit = 4600
        count = [0, 0, 0, 0]
        with open(self.chunk_path + str(chunk_index) + '.json', 'r') as json_raw:
            for lines in json_raw:
                # line_buf = json_chunk_file.readline()
                line_jsonify = json.loads(lines)  # build a dictionary
                helpful_list = line_jsonify['helpful']
                # if hlpfl_list[1] == 0:
                #     hlpfl_score = 0.6
                # else:
                #     hlpfl_score = hlpfl_list[0] / hlpfl_list[1]  # smoothing
                if helpful_list[1] != 0:
                    score_raw = helpful_list[0] / helpful_list[1]
                    score_smooth = score_raw - 1/(1 + helpful_list[1])
                    if score_smooth <= threshold:
                        helpful_score = 0  # not helpful
                        text_0.append(str(line_jsonify['reviewText']))
                        score_0.append(helpful_score)
                        count[0] += 1
                    else:
                        helpful_score = 1  # helpful
                        text_1.append(str(line_jsonify['reviewText']))
                        score_1.append(helpful_score)
                        count[1] += 1
                # elif helpful_list[0] < helpful_list[1] / 2:
                #     helpful_score = 0  # not helpful
                #     text_2.append(str(line_jsonify['reviewText']))
                #     score_2.append(helpful_score)
                #     count[0] += 1
                else:
                    helpful_score = 3  # abnormal cases
                    count[3] += 1

            json_raw.close()
        self.review_helpful_score = score_0 + score_1# + score_2
        self.review_text_list = text_0 + text_1# + text_2
        print(len(self.review_text_list))
        print('Electronics chunk', chunk_index, 'file loaded. Label count: ')
        # count = [0, 0, 0, 0]
        # for item in self.review_helpful_score:
        #     if item == 0:
        #         count[0] += 1
        #     elif item == 1:
        #         count[1] += 1
        #     elif item == 2:
        #         count[2] += 1
        #     else:
        #         count[3] += 1
        print(count)

        with open(os.path.join(DATA_PATH, 'tokenizer.pkl'), 'rb') as f_pkl:
            tkn = pickle.load(f_pkl)
            f_pkl.close()

        with open(self.csv_path + str(chunk_index) + '_bin_th.csv', 'w', newline='') as seq_csv_file:
            csv_writer = csv.writer(seq_csv_file, quoting=csv.QUOTE_MINIMAL)
            print('Total number of reviews: ', len(self.review_text_list))
            for i in range(len(self.review_text_list)):
                single_line = [self.review_helpful_score[i]]
                word_index_seq = tkn.texts_to_sequences(self.review_text_list[i])
                # print(xx)
                if len(word_index_seq) > MAX_LENGTH:
                    word_index_seq = word_index_seq[:MAX_LENGTH]  # sequence cropping
                if i % 10000 == 0:
                    print(i, 'reviews are processed.')
                line_no_bracket = []
                for x in word_index_seq:
                    if x != []:
                        line_no_bracket.append(x[0])
                    else:
                        line_no_bracket.append(0)
                single_line += line_no_bracket
                csv_writer.writerow(single_line)
            seq_csv_file.close()
            print('Chunk', chunk_index, 'File with all tokens saved.')

    def load_train_test_seq(self, chunk_num):

        print('Chunk', chunk_num, 'loading...')
        with open(self.csv_path + str(chunk_num) + '_bin_th.csv', 'r', newline='') as seq_csv_file:
            csv_reader = csv.reader(seq_csv_file)
            total_list = list(csv_reader)
            seq_csv_file.close()

        for i in range(len(total_list)):
            list_line = [int(x) for x in total_list[i][1:]]
            if 80 <= i % 100 < 100:
                self.X_test.append(list_line)
                self.Y_test.append([int(total_list[i][0])])
            else:
                self.X_train.append(list_line)
                self.Y_train.append([int(total_list[i][0])])


class Vocabulary(object):

    def __init__(self):
        self.vocabulary = {}
        self.word_count = 0

    def add_word(self, word):
        if word not in self.vocabulary:
            self.vocabulary[word] = self.word_count
            self.word_count += 1

    def word_indexing(self, word):
        try:
            return self.vocabulary[word]
        except KeyError:
            print('Word not in vocabulary set, build the dictionary again.')


class ReviewRNNModel(object):
    def __init__(self, start_time, model_h5_name):
        self.model = None
        self.model_h5_name = model_h5_name
        self.model_path = ROOT_PATH + self.model_h5_name + '.h5'
        self.hidden_units = 256
        self.batch_size = 128
        self.nb_epoch = 3
        self.nb_cls = 2
        self.max_len = 256  # remove the sequence after 500 words
        self.max_features = 50000  # vocabulary size
        self.start_time_stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(start_time))
        # K.set_image_dim_ordering('th')

    def network_clf(self):

        # embedding_matrix.pkl

        model_h5_file = Path(self.model_path)
        if False:#model_h5_file.is_file():
            self.model = load_model(self.model_path)
            print('Network model loaded. Model Summary:')
        else:
            with open(os.path.join(DATA_PATH, 'embedding_matrix.pkl'), 'rb') as f_pkl:
                embedding_matrix = pickle.load(f_pkl)
                f_pkl.close()
            print(embedding_matrix.shape)
            print('Embedding matrix loaded.')
            with open(os.path.join(DATA_PATH, 'tokenizer.pkl'), 'rb') as f_pkl:
                tkn = pickle.load(f_pkl)
                f_pkl.close()
            print('Tokenizer loaded.')
            print(len(tkn.word_index))
            self.model = Sequential()
            # self.model.add(Embedding(input_dim=len(tkn.word_index) + 1,
            #                          input_length=self.max_len,
            #                          output_dim=1,
            #                          ))
            self.model.add(Embedding(input_dim=len(tkn.word_index)+1,
                                     input_length=self.max_len,
                                     output_dim=100,
                                     weights=[embedding_matrix],
                                     trainable=False))
            self.model.add(LSTM(units=self.hidden_units,
                                     input_shape=(256, 100),
                                     recurrent_dropout=0.1,
                                     dropout=0.1,
                                     # return_sequences=True
                                     ))
            # self.model.add(SimpleRNN(units=100,
            #                          recurrent_dropout=0.1,
            #                          dropout=0.1
            #                          ))
            self.model.add(Dense(50))
            self.model.add(Activation('relu'))
            self.model.add(Dense(1))  # fully connected layer, 10 classes
            self.model.add(Activation('sigmoid'))

            def fscore(y_true, y_pred):

                beta = 1

                # Count positive samples.
                c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
                c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

                # If there are no true samples, fix the F score at 0.
                if c3 == 0:
                    return 0

                # How many selected items are relevant?
                precision = c1 / c2

                # How many relevant items are selected?
                recall = c1 / c3

                # Weight precision and recall together as a single scalar.
                beta2 = beta ** 2
                f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
                return f_score

            def precision(y_true, y_pred):
                # Count positive samples.
                c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
                c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

                # If there are no true samples, fix the F score at 0.
                if c3 == 0:
                    return 0

                # How many selected items are relevant?
                return c1 / c2

            def recall(y_true, y_pred):
                # Count positive samples.
                c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
                c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

                # If there are no true samples, fix the F score at 0.
                if c3 == 0:
                    return 0
                return c1 / c3
            self.model.compile(loss='binary_crossentropy', # 'categorical_crossentropy',
                               optimizer='adadelta',  # RMSprop
                               metrics=['accuracy', fscore, precision, recall])
            self.model.save(self.model_path)

            json_string = self.model.to_json()
            with open(ROOT_PATH + 'result_output/' + self.model_h5_name + '.json', 'w') as f_json:
                f_json.write(json_string)
                f_json.close()
            print('Network definition done. Model Summary:')
        print(self.model.summary())

    def model_fitting(self, X_train, Y_train, X_test, Y_test):
        # CSV logger
        csv_path = os.path.join(ROOT_PATH, 'result_output/' + self.model_h5_name + self.start_time_stamp + '_log.log')
        csv_logger = callbacks.CSVLogger(csv_path, append=True)

        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len, padding='post')
        # Y_train = np_utils.to_categorical(Y_train, self.nb_cls)  # [0/1, 0/1, ..., 0/1]
        print('X train shape:', X_train.shape)
        # print('Y train shape:', Y_train.shape)

        X_test = sequence.pad_sequences(X_test, maxlen=self.max_len, padding='post')
        # Y_test = np_utils.to_categorical(Y_test, self.nb_cls)
        print('X test shape:', X_test.shape)
        # print('Y test shape:', Y_test.shape)

        self.model.fit(X_train, Y_train,
                       batch_size=self.batch_size,
                       epochs=self.nb_epoch,
                       verbose=1,
                       shuffle=True,
                       validation_data=(X_test, Y_test),
                       callbacks=[csv_logger])
        self.model.save(self.model_path)

    def test_validate(self, X_test, y_test):
        model_h5_file = Path(ROOT_PATH + self.model_h5_name)
        if model_h5_file.is_file():
            self.model = load_model(ROOT_PATH + self.model_h5_name)

        X_test = sequence.pad_sequences(X_test, maxlen=self.max_len)
        print('x_test shape:', X_test.shape)
        Y_test = np_utils.to_categorical(y_test, self.nb_cls)

        score = self.model.evaluate(X_test, Y_test, verbose=1, batch_size=self.batch_size)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def knn_method(self, X_train, Y_train, X_test, Y_test):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len, padding='post')
        X_test = sequence.pad_sequences(X_test, maxlen=self.max_len, padding='post')

        Y_train = np.asarray(Y_train)
        Y_test = np.asarray(Y_test)
        Y_train = Y_train.reshape(Y_train.shape[0], )
        Y_test = Y_test.reshape(Y_test.shape[0], )
        print(Y_train.shape)
        print(Y_test.shape)

        print('DEBUG: Machine learning - kNN')
        clf_knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
        clf_knn.fit(X_train, Y_train)
        score = clf_knn.score(X_test, Y_test)
        print('kNN score:', score)
        cv_scores = cross_val_score(clf_knn, X_train, Y_train, cv=5)
        print(cv_scores)
        print('KNN CV score: ' + str(cv_scores.mean()) + ' Std: ' + str(np.std(cv_scores)))

    def svm_method(self, X_train, Y_train, X_test, Y_test):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len, padding='post')
        X_test = sequence.pad_sequences(X_test, maxlen=self.max_len, padding='post')

        Y_train = np.asarray(Y_train)
        Y_test = np.asarray(Y_test)
        Y_train = Y_train.reshape(Y_train.shape[0],)
        Y_test = Y_test.reshape(Y_test.shape[0], )
        print(Y_train.shape)
        print(Y_test.shape)

        print('DEBUG: Machine learning - SVM')
        clf_svm = svm.SVC(kernel='linear', cache_size=2000)
        clf_svm.fit(X_train, Y_train)
        score = clf_svm.score(X_test, Y_test)
        print('SVM score:', score)
        cv_scores = cross_val_score(clf_svm, X_train, Y_train, cv=5)
        print(cv_scores)
        print('SVM CV score: ' + str(cv_scores.mean()) + ' Std: ' + str(np.std(cv_scores)))


class DataPreprocessorReg(DataPreprocessor):

    def chunk_json_raw_to_csv(self, chunk_index):

        self.review_text_list = []
        self.review_helpful_score = []

        with open(self.chunk_path + str(chunk_index) + '.json', 'r') as json_raw:
            for lines in json_raw:
                # line_buf = json_chunk_file.readline()
                line_jsonify = json.loads(lines)  # build a dictionary

                helpful_list = line_jsonify['helpful']

                if helpful_list[1] != 0:
                    self.review_text_list.append(str(line_jsonify['reviewText']))
                    helpful_score = helpful_list[0] / helpful_list[1] - 1/(10+helpful_list[1]) # true score
                    # it is possible to have negative values
                    self.review_helpful_score.append(helpful_score)
                # else:

            json_raw.close()
        print('Electronics chunk', chunk_index, 'file loaded. Label count: ')
        count = [0, 0, 0, 0]
        for item in self.review_helpful_score:
            if item <= 0:
                count[0] += 1
            elif item != 0:
                count[1] += 1
            if item == 1:
                count[2] += 1
            else:
                count[3] += 1
        print(count)

        with open(os.path.join(DATA_PATH, 'tokenizer.pkl'), 'rb') as f_pkl:
            tkn = pickle.load(f_pkl)
            f_pkl.close()

        with open(self.csv_path + str(chunk_index) + '_reg_nz_penalty.csv', 'w', newline='') as seq_csv_file:
            csv_writer = csv.writer(seq_csv_file, quoting=csv.QUOTE_MINIMAL)
            print('Total number of reviews: ', len(self.review_text_list))
            for i in range(len(self.review_text_list)):
                single_line = [self.review_helpful_score[i]]
                word_index_seq = tkn.texts_to_sequences(self.review_text_list[i])
                # print(xx)
                if len(word_index_seq) > MAX_LENGTH:
                    word_index_seq = word_index_seq[:MAX_LENGTH]  # sequence cropping
                if i % 10000 == 0:
                    print(i, 'reviews are processed.')
                line_no_bracket = []
                for x in word_index_seq:
                    if x != []:
                        line_no_bracket.append(x[0])
                    else:
                        line_no_bracket.append(0)
                single_line += line_no_bracket
                csv_writer.writerow(single_line)
            seq_csv_file.close()
            print('Chunk', chunk_index, 'File with all tokens saved.')

    def load_train_test_seq(self, chunk_num):
        # Total_num = 1689188
        # train_count = 0
        # test_count = 0
        # for i in range(Total_num):
        #     if 8000 < i % 10000 < 10000:
        #         test_count += 1
        #     else:
        #         train_count += 1
        # print(train_count, test_count)
        print('Chunk', chunk_num, 'loading...')
        with open(self.csv_path + str(chunk_num) + '_reg_nz_penalty.csv', 'r', newline='') as seq_csv_file:
            csv_reader = csv.reader(seq_csv_file)
            total_list = list(csv_reader)
            seq_csv_file.close()

        for i in range(len(total_list)):
            list_line = [int(x) for x in total_list[i][1:]]
            if 800 <= i % 1000 < 1000:
                self.X_test.append(list_line)
                self.Y_test.append([float(total_list[i][0])])
            else:
                self.X_train.append(list_line)
                self.Y_train.append([float(total_list[i][0])])

        # print(self.Y_train[0:50])


class ReviewRNNModelReg(ReviewRNNModel):

    def network_reg(self):
        # use neural network to do regression
        # embedding_matrix.pkl

        model_h5_file = Path(self.model_path)
        if False:  # model_h5_file.is_file():
            self.model = load_model(self.model_path)
            print('Network model loaded. Model Summary:')
        else:
            with open(os.path.join(DATA_PATH, 'tokenizer.pkl'), 'rb') as f_pkl:
                tkn = pickle.load(f_pkl)
                f_pkl.close()
            print(len(tkn.word_index))
            print('Tokenizer loaded.')
            with open(os.path.join(DATA_PATH, 'embedding_matrix.pkl'), 'rb') as f_pkl:
                embedding_matrix = pickle.load(f_pkl)
                f_pkl.close()
            print(embedding_matrix.shape)
            print('Embedding matrix loaded.')

            self.model = Sequential()
            self.model.add(Embedding(input_dim=len(tkn.word_index) + 1,
                                     input_length=self.max_len,
                                     output_dim=100,
                                     weights=[embedding_matrix],
                                     trainable=False))
            self.model.add(SimpleRNN(units=self.hidden_units,
                                input_shape=(256, 100),
                                recurrent_dropout=0.1,
                                dropout=0.1,
                                return_sequences=True
                                ))
            # self.model.add(SimpleRNN(units=100,
            #                          recurrent_dropout=0.1,
            #                          dropout=0.1
            #                          ))
            # self.model.add(Dense(50))
            self.model.add(Activation('relu'))
            self.model.add(Flatten())
            self.model.add(Dense(1))  # fully connected layer, 10 classes
            self.model.add(Activation('linear'))
            self.model.compile(loss='cosine_proximity',
                               optimizer='adam',  # RMSprop
                               metrics=['cosine_proximity'])
            self.model.save(self.model_path)

            json_string = self.model.to_json()
            with open(ROOT_PATH + 'result_output/' + self.model_h5_name + '.json', 'w') as f_json:
                f_json.write(json_string)
                f_json.close()
            print('Network definition done. Model Summary:')
        print(self.model.summary())

    def model_fitting_reg(self, X_train, Y_train, X_test, Y_test):
        # CSV logger
        csv_path = os.path.join(ROOT_PATH, 'result_output/' + self.model_h5_name + self.start_time_stamp + '_log.log')
        csv_logger = callbacks.CSVLogger(csv_path, append=True)

        # tensorboard initialization
        # tsb_path = './result_output/logs/'
        # if not os.path.exists(tsb_path):
        #     os.makedirs(tsb_path)
        # tsb = callbacks.TensorBoard(log_dir=tsb_path)

        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len, padding='post')
        Y_train = np.asarray(Y_train)
        print('X train shape:', X_train.shape)
        print('Y train shape:', Y_train.shape)

        X_test = sequence.pad_sequences(X_test, maxlen=self.max_len, padding='post')
        Y_test = np.asarray(Y_test)
        print('X test shape:', X_test.shape)
        print('Y test shape:', Y_test.shape)

        self.model.fit(X_train, Y_train,
                       batch_size=self.batch_size,
                       epochs=self.nb_epoch,
                       verbose=1,
                       shuffle=True,
                       validation_data=(X_test, Y_test),
                       callbacks=[csv_logger])
        self.model.save(self.model_path)