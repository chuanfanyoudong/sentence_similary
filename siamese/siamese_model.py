# coding: utf-8
# File: siamese_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23
from __future__ import division
import pickle

from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.regularizers import L1L2

from langconv import *
import sys
import numpy as np
from keras import backend as K, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Lambda, Bidirectional, Conv2D, Conv1D, Reshape, dot, \
    MaxPooling1D, GRU, CuDNNLSTM, GlobalAveragePooling1D, Concatenate, GlobalMaxPooling1D
import os
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class LR_Updater(Callback):
    '''
    Abstract class where all Learning Rate updaters inherit from. (e.g., CircularLR)
    Calculates and updates new learning rate and momentum at the end of each batch.
    Have to be extended.
    '''
    def __init__(self, init_lrs):
        self.init_lrs = init_lrs
        super(LR_Updater, self).__init__()

    def on_train_begin(self, logs=None):
        self.update_lr()

    def on_batch_end(self, batch, logs=None):
        self.update_lr()

    def update_lr(self):
        # cur_lrs = K.get_value(self.model.optimizer.lr)
        new_lrs = self.calc_lr(self.init_lrs)
        K.set_value(self.model.optimizer.lr, new_lrs)

    def calc_lr(self, init_lrs): raise NotImplementedError


class CircularLR(LR_Updater):
    '''
    A learning rate updater that implements the CircularLearningRate (CLR) scheme.
    Learning rate is increased then decreased linearly.
    '''
    def __init__(self, init_lrs, nb, div=4, cut_div=8, on_cycle_end=None):
        self.nb,self.div,self.cut_div,self.on_cycle_end = nb,div,cut_div,on_cycle_end
        super(CircularLR, self).__init__(init_lrs)

    def on_train_begin(self, logs=None):
        self.cycle_iter,self.cycle_count=0,0
        super(CircularLR, self).on_train_begin()

    def calc_lr(self, init_lrs):
        cut_pt = self.nb//self.cut_div
        if self.cycle_iter>cut_pt:
            pct = 1 - (self.cycle_iter - cut_pt)/(self.nb - cut_pt)
        else: pct = self.cycle_iter/cut_pt
        res = init_lrs * (1 + pct*(self.div-1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res


class SiameseNetwork:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # self.train_path = os.path.join(cur, 'data/train_final.txt')
        # self.train_path = os.path.join(cur, 'data/train')
        self.train_path = os.path.join(cur, 'data/ant_all.text')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_siamese_model.h5')
        self.datas, self.word_dict = self.build_data()
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 500
        self.BATCH_SIZE = 256
        self.NUM_CLASSES = 2
        self.VOCAB_SIZE = len(self.word_dict)
        self.LIMIT_RATE = 0.95
        self.TIME_STAMPS = self.select_best_length()
        self.embedding_matrix = self.build_embedding_matrix()

    def process_lint(self, line):

        line = Converter('zh-hans').convert(line.decode("utf-8"))
        line = "".join(line.strip().split(" "))
        line = u"花呗".join(line.strip().split(u"huabei"))
        line = u"*".join(line.strip().split(u"***"))
        line = u"花呗".join(line.strip().split(u"花被"))
        line = u"借呗".join(line.strip().split(u"借被"))
        line = u"花呗".join(line.strip().split(u"蚂蚁花呗"))
        line = u"借呗".join(line.strip().split(u"蚂蚁借呗"))
        return line

    def select_best_length(self):
        len_list = []
        max_length = 0
        cover_rate = 0.0
        for line in open(self.train_path):
            line = self.process_lint(line)
            line = line.strip().split('\t')
            if not line:
                continue
            sent = line[1]
            # sent_len = len(sent)
            sent_len = len(sent)
            len_list.append(sent_len)
        all_sent = len(len_list)
        sum_length = 0
        len_dict = Counter(len_list).most_common()
        len_dict = sorted(len_dict, key=lambda x:x[0])
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


    def build_data(self):
        sample_x = []
        sample_y = []
        sample_x_left = []
        sample_x_right = []
        vocabs = {'UNK'}
        for line in open(self.train_path):
            line = self.process_lint(line)
            line = line.rstrip().split('\t')
            # line = line.rstrip().split('\t')
            if not line:
                continue
            sent_left = line[1]
            sent_right = line[2]
            label = line[3]
            sample_x_left.append([char for char in sent_left if char])
            sample_x_right.append([char for char in sent_right if char])
            sample_y.append(label)
            for char in [char for char in sent_left + sent_right if char]:
                vocabs.add(char)
        print(len(sample_x_left), len(sample_x_right))
        sample_x = [sample_x_left, sample_x_right]
        datas = [sample_x, sample_y]
        word_dict = {wd:index for index, wd in enumerate(list(vocabs))}
        writer = open("./model/word_dict.pkl", "wb")
        pickle.dump(word_dict, writer, protocol=2)
        # self.write_file(list(vocabs), self.vocab_path)
        return datas, word_dict

    def modify_data(self):
        sample_x = self.datas[0]
        sample_y = self.datas[1]
        sample_x_left = sample_x[0]
        sample_x_right = sample_x[1]
        left_x_train = [[self.word_dict[char] for char in data] for data in sample_x_left]
        right_x_train = [[self.word_dict[char] for char in data] for data in sample_x_right]
        y_train = [int(i) for i in sample_y]
        left_x_train = pad_sequences(left_x_train, self.TIME_STAMPS)
        right_x_train = pad_sequences(right_x_train, self.TIME_STAMPS)
        y_train = np.expand_dims(y_train, 2)
        return left_x_train, right_x_train, y_train


    def write_file(self, wordlist, filepath):
        with open(filepath, 'w') as f:
            for line in wordlist:
                f.write(line.encode("utf-8") + "\n")
            # f.write('\n'.join(wordlist))


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
        # print embeddings_dict
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict


    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.random.uniform(-0.01, 0.01, (self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        # np.random.uniform(-scale, scale, [1, embedd_dim])
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                # print word
                embedding_matrix[i] = embedding_vector
            else:
                print word
        return embedding_matrix


    def exponent_neg_manhattan_distance(self, inputX):
        (sent_left, sent_right) = inputX
        return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))


    def euclidean_distance(self, inputX):
        (sent_left, sent_right) = inputX
        sum_square = K.sum(K.square(sent_left - sent_right), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))



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
        model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape = input_shape)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(32)))
        return model

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
        # loss = Lambda(lambda x: K.relu(x))(right_cos)
        # distance = Lambda(lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]),
        #                   output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
        # distance = Lambda(self.exponent_neg_manhattan_distance, name = "xiaosa")([left_output, right_output])
        ipt = Input(shape=(self.TIME_STAMPS, self.EMBEDDING_DIM))
        dropout_rate = 0.5
        x = Dropout(dropout_rate, )(ipt)
        for i, hidden_length in enumerate([64,64,64]):
            # x = Bidirectional(CuDNNLSTM(hidden_length, return_sequences=(i!=len(n_hidden)-1), kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)
            x = Bidirectional(
                LSTM(hidden_length, return_sequences=True, kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)

        # v3 卷及网络特征层
        x = Conv1D(64, kernel_size=2, strides=1, padding="valid", kernel_initializer="he_uniform")(x)
        x_p1 = GlobalAveragePooling1D()(x)
        x_p2 = GlobalMaxPooling1D()(x)
        x = Concatenate()([x_p1, x_p2])
        shared_lstm = Model(inputs=ipt, outputs=x)

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # 距离函数 exponent_neg_manhattan_distance
        malstm_distance = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)),
                                 output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        model = Model([left_input, right_input], [malstm_distance])
        model = Model([left_input, right_input], right_cos)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001, beta_1=0.8),
                      metrics=[self.f1])
        model.summary()

        return model

    def siamese(self, pretrained_embedding=None,
                input_length= 25,
                w2v_length=300,
                n_hidden=[64, 64, 64]):

        input_length = self.TIME_STAMPS
                # 输入层
        left_input = Input(shape=(input_length,), dtype='int32')
        right_input = Input(shape=(input_length,), dtype='int32')

        # 对句子embedding
        encoded_left = pretrained_embedding(left_input)
        encoded_right = pretrained_embedding(right_input)

        # 两个LSTM共享参数
        # # v1 一层lstm
        # shared_lstm = CuDNNLSTM(n_hidden)

        # # v2 带drop和正则化的多层lstm
        ipt = Input(shape=(input_length, w2v_length))
        dropout_rate = 0.5
        x = Dropout(dropout_rate, )(ipt)
        for i, hidden_length in enumerate(n_hidden):
            # x = Bidirectional(CuDNNLSTM(hidden_length, return_sequences=(i!=len(n_hidden)-1), kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)
            x = Bidirectional(
                CuDNNLSTM(hidden_length, return_sequences=True, kernel_regularizer=L1L2(l1=0.01, l2=0.01)))(x)

        # v3 卷及网络特征层
        x = Conv1D(64, kernel_size=2, strides=1, padding="valid", kernel_initializer="he_uniform")(x)
        x_p1 = GlobalAveragePooling1D()(x)
        x_p2 = GlobalMaxPooling1D()(x)
        x = Concatenate()([x_p1, x_p2])
        shared_lstm = Model(inputs=ipt, outputs=x)

        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        # 距离函数 exponent_neg_manhattan_distance
        malstm_distance = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)),
                                 output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        model = Model([left_input, right_input], [malstm_distance])

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

    def train_model(self):
        left_x_train, right_x_train, y_train = self.modify_data()
        print(y_train)
        init_lrs = 0.001
        clr_div, cut_div = 10, 8
        # batch_num = (train_x[0].shape[0] - 1) // train_batch_size + 1
        # cycle_len = 1
        # total_iterators = batch_num * cycle_len
        circular_lr = CircularLR(init_lrs, 60,  on_cycle_end=None, div=clr_div, cut_div=cut_div)
        callbacks = [circular_lr]


        model = self.bilstm_siamese_model()
        print "迭代次数"
        print self.EPOCHS
        history = model.fit(
                              x=[left_x_train, right_x_train],
                              y=y_train,
                              validation_split=0.2,
                              batch_size=self.BATCH_SIZE,
                              epochs=self.EPOCHS,
                               callbacks = callbacks
                            )
        model.save(self.model_path)


    # def predict(self):
    #     left_x_train, right_x_train, y_train= self.modify_data()
    #     # left_x_train, right_x_train, y_train = left_x_train[:100], right_x_train[:100], y_train[:100]
    #     model = self.bilstm_siamese_model()
    #     model.load_weights(self.model_path)
    #     result = model.predict([left_x_test, right_x_test])
    #     result = [1 if i[0] > 0.5 else 0 for i in result]
    #     y_test = [i[0] for i in y_test]
    #     f1 = f1_score(y_test, result, average='macro')
    #     p = precision_score(y_test, result, average='macro')
    #     r = recall_score(y_test, result, average='macro')
    #     a = accuracy_score(y_test, result)
    #     print a
    #     print p
    #     print r
    #     print f1
    #     return a, p, r, f1




handler = SiameseNetwork()
handler.train_model()
# handler.predict()

