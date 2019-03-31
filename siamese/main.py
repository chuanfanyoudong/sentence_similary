# encoding: utf-8
# coding=utf-8
# -*- coding: UTF-8 -*-
# File: siamese_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23
import pickle
import sys

import numpy as np
from langconv import *
from keras import backend as K, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Lambda, Bidirectional, Conv1D, MaxPooling1D, Reshape, dot
import os
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SiameseNetwork:
    def __init__(self, inputfile):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # self.train_path = os.path.join(cur, 'data/train.txt')
        self.train_path = os.path.join(inputfile)
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_siamese_model.h5')
        self.datas, self.word_dict = self.build_data()
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 20
        self.BATCH_SIZE = 512
        self.NUM_CLASSES = 2
        self.VOCAB_SIZE = len(self.word_dict)
        self.LIMIT_RATE = 0.95
        self.TIME_STAMPS = 25
        self.embedding_matrix = self.build_embedding_matrix()

    '''根据样本长度,选择最佳的样本max-length'''
    def select_best_length(self):
        len_list = []
        max_length = 0
        cover_rate = 0.0
        for line in open(self.train_path):
            line = self.process_lint(line)
            line = line.strip().split('	')
            if not line:
                continue
            sent = line[1]
            sent_len = len(sent)
            len_list.append(sent_len)
        all_sent = len(len_list)
        sum_length = 0
        len_dict = Counter(len_list).most_common()
        len_dict = sorted(len_dict, key=lambda x: x[0])
        for i in len_dict:
            sum_length += i[1]*i[0]
        average_length = sum_length/all_sent
        for i in len_dict:
            rate = i[1]/all_sent
            cover_rate += rate
            if cover_rate >= self.LIMIT_RATE:
                max_length = i[0]
                break
        print('average_length:', average_length)
        print('max_length:', max_length)
        return max_length

    '''构造数据集'''
    def build_data(self):
        sample_x = []
        sample_y = []
        sample_x_left = []
        sample_x_right = []
        vocabs = {'UNK'}
        for line in open(self.train_path):
            line = self.process_lint(line)
            line = line.rstrip().split('\t')
            if not line:
                continue
            sent_left = line[1]
            sent_right = line[2]
            sample_x_left.append([char for char in sent_left if char])
            sample_x_right.append([char for char in sent_right if char])
            for char in [char for char in sent_left + sent_right if char]:
                vocabs.add(char)
        print(len(sample_x_left), len(sample_x_right))
        sample_x = [sample_x_left, sample_x_right]
        datas = [sample_x, sample_y]
        writer = open("./model/word_dict.pkl", "rb")
        word_dict = pickle.load(writer)
        # self.write_file(list(vocabs), self.vocab_path)
        return datas, word_dict



    '''将数据转换成keras所需的格式'''
    def modify_data(self):
        sample_x = self.datas[0]
        sample_x_left = sample_x[0]
        sample_x_right = sample_x[1]
        left_x_train = [[self.word_dict[char] if char in self.word_dict else self.word_dict["UNK"] for char in data ] for data in sample_x_left]
        right_x_train = [[self.word_dict[char] if char in self.word_dict else self.word_dict["UNK"] for char in data] for data in sample_x_right]
        left_x_train = pad_sequences(left_x_train, self.TIME_STAMPS)
        right_x_train = pad_sequences(right_x_train, self.TIME_STAMPS)
        return left_x_train, right_x_train


    def process_lint(self, line):
        line = Converter('zh-hans').convert(line.decode("utf-8"))
        line = "".join(line.strip().split(" "))
        line = u"花呗".join(line.strip().split(u"huabei"))
        line = u"花呗".join(line.strip().split(u"花被"))
        return line


    '''保存字典文件'''
    def write_file(self, wordlist, filepath):
        with open(filepath, 'w+') as f:
            f.write('\n'.join(wordlist))

    '''加载预训练词向量'''
    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r') as f:
            for line in f:
                values = line.decode("utf-8").strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    '''加载词向量矩阵'''
    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''
    def exponent_neg_manhattan_distance(self, inputX):
        (sent_left, sent_right) = inputX
        return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

    '''基于欧式距离的字符串相似度计算'''
    def euclidean_distance(self, sent_left, sent_right):
        sum_square = K.sum(K.square(sent_left - sent_right), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))


    '''搭建编码层网络,用于权重共享'''


    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
        # input = Bidirectional(GRU(128))(input)
        lstm1 = Dropout(0.5)(lstm1)
        # lstm2 = Bidirectional(LSTM(32))(lstm1)
        # lstm2 = Dropout(0.5)(lstm2)
        final1 = Conv1D(64, kernel_size=(2), strides=1, activation='relu')(lstm1)
        # final2 = Conv1D(64, kernel_size=(3), strides=1, activation='relu')(lstm1)
        # final3 = Conv1D(64, kernel_size=(4), strides=1, activation='relu')(lstm1)
        # final4 = Conv1D(64, kernel_size=(5), strides=1, activation='relu')(lstm1)
        # final1 = MaxPooling1D(pool_size=(final1.shape[1].value), strides=None, padding='valid')(final1)
        # final2 = MaxPooling1D(pool_size=(final2.shape[1].value), strides=None, padding='valid')(final2)
        # final3 = MaxPooling1D(pool_size=(final3.shape[1].value), strides=None, padding='valid')(final3)
        # final4 = MaxPooling1D(pool_size=(final4.shape[1].value), strides=None, padding='valid')(final4)
        # final = Lambda(lambda x: K.concatenate([x[0], x[1], x[2], x[3]], axis=2))([final1, final2, final3, final4])
        final = Lambda(lambda x: K.sum(x, axis= 1))(final1)
        # final = Reshape((64,))(final1)
        return Model(input, final)

    def create_model(self, input_shape):
        # 作为 Sequential 模型的第一层
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(32)))
        return model

    def f1(self, y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.
            Only computes a batch-wise average of recall.
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.
            Only computes a batch-wise average of precision.
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    '''搭建网络'''
    def bilstm_siamese_model(self):
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    )

        left_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        right_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)
        shared_lstm = self.create_base_network(input_shape=(self.TIME_STAMPS, self.EMBEDDING_DIM))
        # shared_lstm = self.create_model(input_shape=(self.TIME_STAMPS, self.EMBEDDING_DIM))
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)
        right_cos = dot([left_output, right_output], -1, normalize=True)
        loss = Lambda(lambda x: K.relu(x))(right_cos)
        # distance = Lambda(lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]),
        #                   output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        # distance = Lambda(self.exponent_neg_manhattan_distance, name = "xiaosa")([left_output, right_output])

        model = Model([left_input, right_input], loss)
        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=[self.f1])
        model.summary()

        return model

    def predict(self, inpath, outpath):
        self.train_path = inpath
        left_x_train, right_x_train = self.modify_data()
        # left_x_train, right_x_train = left_x_train[:100], right_x_train[:100]
        model = self.bilstm_siamese_model()
        model.load_weights(self.model_path)
        result = model.predict([left_x_train, right_x_train])
        result = [1 if i[0] > 0.5 else 0 for i in result]
        # print result
        with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
            i = -1
            for line in fin:
                # line = self.process_lint(line)
                i += 1
                lineno = line.strip().split('\t')[0]
                # print lineno
                fout.write(lineno + '\t' + str(result[i]) + '\n')


def process(inpath, outpath):
    handler = SiameseNetwork(inpath)
    # handler.predict(sys.argv[1], sys.argv[2])
    handler.predict(inpath, outpath)

if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])

