#!/usr/bin/env python3
# coding: utf-8
# File: siamese_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23
import pickle
import sys
import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Lambda, Bidirectional
import os
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SiameseNetwork:
    def __init__(self, input_path):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # self.train_path = os.path.join(cur, 'data/train.txt')
        self.train_path = os.path.join(input_path)
        self.LIMIT_RATE = 0.95
        self.TIME_STAMPS = 20
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_siamese_model.h5')
        self.datas, self.word_dict = self.build_data()
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 2
        self.BATCH_SIZE = 512
        self.NUM_CLASSES = 2
        self.VOCAB_SIZE = len(self.word_dict)
        self.embedding_matrix = self.build_embedding_matrix()

    def select_best_length(self):
        len_list = []
        max_length = 0
        cover_rate = 0.0
        for line in open(self.train_path):
            line = line.strip().split('	')
            if not line:
                continue
            sent = line[1]
            sent_len = len(sent)
            len_list.append(sent_len)
        all_sent = len(len_list)
        sum_length = 0
        len_dict = Counter(len_list).most_common()
        for i in len_dict:
            sum_length += i[1]*i[0]
        average_length = sum_length/all_sent
        for i in len_dict:
            rate = i[1]/all_sent
            cover_rate += rate
            if cover_rate >= self.LIMIT_RATE:
                max_length = i[0]
                break

        return max_length

    def build_data(self):
        sample_x = []
        sample_y = []
        sample_x_left = []
        sample_x_right = []
        vocabs = {'UNK'}
        for line in open(self.train_path):
            line = line.decode("utf-8").rstrip().split('\t')
            if not line:
                continue
            sent_left = line[1]
            sent_right = line[2]
            sample_x_left.append([char for char in sent_left if char])
            sample_x_right.append([char for char in sent_right if char])
            for char in [char for char in sent_left + sent_right if char]:
                vocabs.add(char)
        sample_x = [sample_x_left, sample_x_right]
        datas = [sample_x, sample_y]
        writer = open("./model/word_dict.pkl", "rb")
        word_dict = pickle.load(writer)
        # self.write_file(list(vocabs), self.vocab_path)
        return datas, word_dict

    def modify_data(self):
        sample_x = self.datas[0]
        sample_x_left = sample_x[0]
        sample_x_right = sample_x[1]
        left_x_train = [[self.word_dict[char] for char in data] for data in sample_x_left]
        right_x_train = [[self.word_dict[char] for char in data] for data in sample_x_right]
        left_x_train = pad_sequences(left_x_train, self.TIME_STAMPS)
        right_x_train = pad_sequences(right_x_train, self.TIME_STAMPS)
        return left_x_train, right_x_train

    def write_file(self, wordlist, filepath):
        with open(filepath, 'w+') as f:
            f.write('\n'.join(wordlist))

    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def exponent_neg_manhattan_distance(self, inputX):
        (sent_left, sent_right) = inputX
        return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

    def euclidean_distance(self, sent_left, sent_right):
        sum_square = K.sum(K.square(sent_left - sent_right), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
        lstm1 = Dropout(0.5)(lstm1)
        lstm2 = Bidirectional(LSTM(32))(lstm1)
        lstm2 = Dropout(0.5)(lstm2)
        return Model(input, lstm2)

    def bilstm_siamese_model(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)

        left_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        right_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')

        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        shared_lstm = self.create_base_network(input_shape=(self.TIME_STAMPS, self.EMBEDDING_DIM))
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # distance = Lambda(lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]),
        #                   output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        distance = Lambda(self.exponent_neg_manhattan_distance, name = "xiaosa")([left_output, right_output])

        model = Model([left_input, right_input], distance)
        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
        model.summary()

        return model

    def predict(self, inpath, outpath):
        self.train_path = inpath
        left_x_train, right_x_train = self.modify_data()
        left_x_train, right_x_train = left_x_train[:100], right_x_train[:100]
        model = self.bilstm_siamese_model()
        model.load_weights(self.model_path)
        result = model.predict([left_x_train, right_x_train])
        with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
            i = -1
            for line in fin:
                i += 1
                lineno, sen1, sen2 = line.strip().split('\t')
                fout.write(lineno + '\t' + str(int(2 * result[i][0])) + '\n')

def process(inpath, outpath):
    handler = SiameseNetwork(inpath)
    # handler.predict(sys.argv[1], sys.argv[2])
    handler.predict(inpath, outpath)

if __name__ == '__main__':
    # process(sys.argv[1], sys.argv[2])
    process("data/test", "data/re")

