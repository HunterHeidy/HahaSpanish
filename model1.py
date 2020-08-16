## some config values 
embed_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras import backend
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate
from keras.callbacks import *

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
# from sklearn.model_selection import KFold

from keras.initializers import *
from keras.layers import *
from keras.models import *
from sklearn.model_selection import train_test_split, StratifiedKFold
   # X_train, X_test, y_train, y_test = train_test_split(df['text'], df['is_humor'], test_size=0.33, random_state=42)
# from capsule_layer import CategoryCap, PrimaryCap, Length, Mask

# from keras_wc_embd import get_embedding_layer
# from keras_wc_embd import get_dicts_generator
# from keras_wc_embd import get_embedding_layer, get_embedding_weights_from_file

import tensorflow as tf

from keras_targeted_dropout import TargetedDropout

from sklearn.model_selection import train_test_split

from keras.utils.vis_utils import plot_model

import keras.preprocessing.text as T
from Modelpre import CyclicLR,Attention,HAN_AttLayer,Capsule, KMaxPooling,AttLayer

def load_and_prec():
    df = pd.read_csv("data/cleaned_data_train_haha.csv")

    # test_df = pd.read_csv("/home/bin_lab/桌面/task6/data/cleaned_trial_data.csv")
    # dev_df = pd.read_csv("/home/bin_lab/桌面/task6/data/cleaned_data_testset-taska.csv")
    # trial_df = pd.read_csv("/home/bin_lab/桌面/task6/data/cleaned_trial_data.csv")
    # train_df = pd.read_csv("/home/bin_lab/桌面/task6/cleaned_train_data.csv")
    # test_df = pd.read_csv("/home/bin_lab/桌面/task6/cleaned_trial_data.csv")


    # train_df = pd.read_csv("/home/bin_lab/桌面/task9A/data/Subtask-A-master/cleaned_train_data_9.csv")
    # test_df = pd.read_csv("/home/bin_lab/桌面/task9A/data/Subtask-A-master/SubtaskA_Trial_Test.csv")

    # print("Train shape : ",df.shape)
    # print("Test shape : ",test_df.shape)
    # print("Dev shape : ",dev_df.shape)
    # print("Trial shape : ",trial_df.shape)
    
    ## fill up the missing values
    train_X = df["text"].fillna("_##_").values
    # test_X = test_df["text"].fillna("_##_").values#
    # dev_X = dev_df["text"].fillna("_##_").values
    # trial_X = trial_df["text"].fillna("_##_").values


    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X_ = tokenizer.texts_to_sequences(train_X)
    # test_X_ = tokenizer.texts_to_sequences(test_X)
    # dev_X_ = tokenizer.texts_to_sequences(dev_X)
    # trial_X_ = tokenizer.texts_to_sequences(trial_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X_, maxlen=maxlen)
    # test_X = pad_sequences(test_X_, maxlen=maxlen)
    # dev_X = pad_sequences(dev_X_, maxlen=maxlen)
    # trial_X = pad_sequences(trial_X_, maxlen=maxlen)

    ## Get the target values
    train_y = df['is_humor'].values
    # test_y = test_df['label1'].values
    # train_y = train_df['label1'].values
    # test_y = test_df['label1'].values
    
    #shuffling the data
    np.random.seed(218)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    # print(df['text'].head(3))
    # print(train_X[3,:])
    # print(train_X)
    # print(df.shape)
    # exit()
    train_id =df["id"]
    # test_id = X_test['id']
    # dev_id = dev_df['id']
    # trial_id = trial_df['id']

    sequence_train = df["text"].fillna("_##_").values.astype(str).tolist()
    # print(sequence_train)

    from nltk.tokenize import WordPunctTokenizer

    def wordtokenizer(sentence):
        words=WordPunctTokenizer().tokenize(sentence)
        return words
    sentence_pairs=[]
    for pair in sequence_train:
        sentence=wordtokenizer(pair)
        sentence_pairs.append(sentence)

    # sentence_pairs = [
    # ['All', 'work', 'and', 'no', 'play'],
    # ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
    # ]
    
    
    return train_X, train_y, tokenizer.word_index, train_id

def load_fasttext(word_index):    
    # EMBEDDING_FILE = '/home/bin_lab/桌面/n2c2-1/data/crawl-300d-2M.vec'
    # def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    # embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    EMBEDDING_FILE = 'data/fasttext-spanish/cc.es.300.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    #embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='gb18030'))

    # EMBEDDING_FILE = '/media/bin_lab/C4F6073207B3A949/Linux/data/glove.840B.300d.txt'
    # def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE,encoding='utf-8') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index)+1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def f1(y_true, y_pred):
    '''
    metric from here 
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def model_HAN(embedding_matrix):
    
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    y = Bidirectional(GRU(40, return_sequences=True))(x)
    
    atten_1 = HAN_AttLayer()(x) # skip connect
    atten_2 = HAN_AttLayer()(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    # conc = Dropout(0.1)(conc)
    conc = TargetedDropout(drop_rate=0.5, target_rate=0.5)(conc)
    outp = Dense(1, activation="sigmoid")(conc)    

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[f1])
    
    return model

def model_lstm_HAN(embedding_matrix):
    
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = LSTM(40, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)(x)
    x = Dropout(0.25)(x)
    attention = HAN_AttLayer()(x)
    fc = Dense(256, activation='relu')(attention)
    fc = Dropout(0.25)(fc)
    fc = BatchNormalization()(fc)
    
    outp = Dense(1, activation="sigmoid")(fc)    

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
    
    return model

def capsulenet_model(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False):
    K.clear_session()       
    inp = Input(shape=(max_sequence_length,))
    x = Embedding(num_words, embedding_dim, weights=[embeddings], trainable=False)(inp)
    x = SpatialDropout1D(rate=0.2)(x)
    x = Bidirectional(GRU(40, return_sequences=True))(x)   #, kernel_initializer=glorot_normal(seed=1230), recurrent_initializer=orthogonal(gain=1.0, seed=1000)

    # x = Bidirectional(LSTM(40, return_sequences=True))(x)
    x = Capsule(num_capsule=10, dim_capsule=16, routings=4, share_weights=True)(x)
    x = Flatten()(x)
    # avg_pool = GlobalAveragePooling1D()(x)
    # x = GlobalMaxPooling1D()(x)
    
    # x = concatenate([flatt, avg_pool])

    x = Dense(16, activation="relu", kernel_initializer=glorot_normal(seed=1230))(x)
    # x = Dropout(0.5)(x)
    x = TargetedDropout(drop_rate=0.5, target_rate=0.5)(x)
    x = BatchNormalization()(x)

    x = Dense(labels_index, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(),metrics=[f1])#Adam()
    model.summary()
    return model

def RCNN_Net(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False, extra_conv=True):
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=trainable)

    # 模型共有三个输入，分别是左词，右词和中心词
    document = Input(shape = (None, ), dtype = "int32")
    left_context = Input(shape = (None, ), dtype = "int32")
    right_context = Input(shape = (None, ), dtype = "int32")

    # 构建词向量
    doc_embedding = embedding_layer(document)
    l_embedding = embedding_layer(left_context)
    r_embedding = embedding_layer(right_context)

    # 分别对应文中的公式(1)-(7)
    forward = GRU(300, return_sequences = True)(l_embedding) # 等式(1)
    # 等式(2)
    backward = GRU(300, return_sequences = True, go_backwards = True)(r_embedding) 
    together = concatenate([forward, doc_embedding, backward], axis = 2) # 等式(3)

    semantic = TimeDistributed(Dense(150, activation = "tanh"))(together) # 等式(4)
    # x1=Dropout(0.5)(semantic)
    # 等式(5)
    pool_rnn = Lambda(lambda x: backend.max(x, axis = 1), output_shape = (150, ))(semantic)
    x2=Dropout(0.5)(pool_rnn)
    output = Dense(labels_index, activation = "sigmoid")(x2) # 等式(6)和(7)
    model = Model(inputs = [document, left_context, right_context], outputs = output)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=[f1])
    model.summary()

    return model

def rnncnn_model(embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(40, return_sequences=True, recurrent_dropout=0.1))(x)
    x = Conv1D(60, kernel_size=3, padding='valid', activation='relu', strides=1)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    y = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    y = SpatialDropout1D(0.2)(x)
    y = Bidirectional(GRU(40, return_sequences=True, recurrent_dropout=0.1))(y)
    y = Conv1D(120, kernel_size=3, padding='valid', activation='relu', strides=1)(y)
    avg_pool2 = GlobalAveragePooling1D()(y)
    max_pool2 = GlobalMaxPooling1D()(y)
    conc = concatenate([avg_pool, max_pool, avg_pool2, max_pool2])
    outp = Dense(1, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=[f1])

    return model

def kmax_textcnn_model(embedding_matrix):
    filter_nums = 180
    drop = 0.6

    inp = Input(shape=(maxlen, ))
    embedded_sequences = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)

    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_3 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)

    # conv_0 = Conv1D(filter_nums / 2, 1, kernel_initializer="normal", padding="valid", activation="relu")(conv_0)
    # conv_1 = Conv1D(filter_nums / 2, 2, kernel_initializer="normal", padding="valid", activation="relu")(conv_1)
    # conv_2 = Conv1D(filter_nums / 2, 3, strides=2, kernel_initializer="normal", padding="valid", activation="relu")(conv_2)

    maxpool_0 = KMaxPooling(k=3)(conv_0)
    maxpool_1 = KMaxPooling(k=3)(conv_1)
    maxpool_2 = KMaxPooling(k=3)(conv_2)
    maxpool_3 = KMaxPooling(k=3)(conv_3)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
    output = Dropout(drop)(merged_tensor)
    # output = TargetedDropout(drop_rate=0.5, target_rate=0.5)(merged_tensor)
    output = Dense(units=144, activation='relu')(output)
    output = Dense(units=1, activation='sigmoid')(output)

    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=[f1])

    model.summary()
    # plot_model(model, to_file="model.png", show_shapes=True)

    return model

def pooled_gru_model(embedding_matrix):
    inp = Input(shape=(maxlen, ))
    embedded_sequences = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    x = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    merged = concatenate([avg_pool, max_pool])
    merged = Dropout(0.1)(merged)
    outp = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=[f1])
    model.summary()

    return model

def lstm_conv_model(embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.35)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation='sigmoid')(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=[f1])
    model.summary()

    return model

def gru128_model(embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(128, dropout=0.3, recurrent_dropout=0.5, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    outp = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  # optimizer='rmsprop',
                  optimizer=Adam(),
                  metrics=[f1])

    model.summary()

    return model

filter_sizes = [1,2,3,5]
num_filters = 36
from keras.layers import Conv1D, MaxPool1D, BatchNormalization
def inceptioncnn_model(embedding_matrix):    
    inp = Input(shape=(maxlen, ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)
    #x = Reshape((maxlen, embed_size, 1))(x)
    
    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]),
                                 kernel_initializer='he_normal', activation='elu')(x)
    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]),
                                 kernel_initializer='he_normal', activation='elu')(x)
    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]), 
                                 kernel_initializer='he_normal', activation='elu')(x)
    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]),
                                 kernel_initializer='he_normal', activation='elu')(x)
    
    maxpool_0 = MaxPool1D(pool_size=(maxlen - filter_sizes[0] + 1))(conv_0)
    maxpool_1 = MaxPool1D(pool_size=(maxlen - filter_sizes[1] + 1))(conv_1)
    maxpool_2 = MaxPool1D(pool_size=(maxlen - filter_sizes[2] + 1))(conv_2)
    maxpool_3 = MaxPool1D(pool_size=(maxlen - filter_sizes[3] + 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = BatchNormalization()(z)
        
    outp = Dense(1, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[f1])

    return model

filter_sizes = [1,2,3,5]
num_filters = 42

def text2dCNN_model(embedding_matrix):    
    inp = Input(shape=(maxlen, ))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
#    x = SpatialDropout1D(0.4)(x)
    x = Reshape((maxlen, embed_size, 1))(x)
    
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size),
                                 kernel_initializer='he_normal', activation='tanh')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size),
                                 kernel_initializer='he_normal', activation='tanh')(x)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), 
                                 kernel_initializer='he_normal', activation='tanh')(x)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size),
                                 kernel_initializer='he_normal', activation='tanh')(x)
    
    maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = Dropout(0.1)(z)
        
    outp = Dense(1, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[f1])

    return model
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = inputs.shape[1].value
    SINGLE_ATTENTION_VECTOR = False
    
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def lstm_att_block_model(embedding_matrix):
    inp = Input(shape=(maxlen, ))
    embed = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = embed
    
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    x = attention_3d_block(x)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    x = AttLayer(maxlen)(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    outp = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[f1])    

    return model

def gru_att_block_model(embedding_matrix):
    inp = Input(shape=(maxlen, ))
    embed = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = embed
    
    x = Bidirectional(GRU(40, return_sequences=True))(x)
    x = attention_3d_block(x)
    x = Bidirectional(GRU(40, return_sequences=True))(x)
    x = AttLayer(maxlen)(x)
    
    
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    outp = Dense(1, activation="sigmoid")(x)
    # outp = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[f1])    

    return model

def gru_att_model(embedding_matrix):
    inp = Input(shape=(maxlen, ))
    embed = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = embed
    
    x0 = Bidirectional(GRU(128, return_sequences=True))(x)
    x1 = attention_3d_block(x0)
    x2 = Bidirectional(GRU(128, return_sequences=True))(x1)
    x3 = Add()([x0, x2])
    x4 = Bidirectional(GRU(64, return_sequences=True))(x3)
    x5 = AttLayer(64)(x4)
    #x5 = Capsule(num_capsule=5, dim_capsule=32, routings=5, share_weights=True)(x4)
    
    x = Dropout(0.3)(x5)
    x = Dense(128, activation='relu')(x)
    outp = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[f1])    

    return model

def model_lstm_atten_1(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def model_gru_srk_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(GRU(40, return_sequences=True))(x)
    x = Attention(maxlen)(x) # New
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model    
    
def model_lstm_du(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(GRU(40, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
############################# Train and predict ###########################
def rcnn_train_pred(model, train_X, train_y, val_X, val_y, epochs=2, callback=None):
    for e in range(epochs):
        model.fit([train_X, x_train_left, x_train_right], train_y, batch_size=512, epochs=1, validation_data=([val_X, x_val_left, x_val_right], val_y), callbacks = callback, verbose=0)
        
        #model.load_weights(filepath)
        pred_val_y = model.predict([val_X, x_val_left, x_val_right], batch_size=1024, verbose=0)

        best_score = metrics.f1_score(val_y, (pred_val_y > 0.33).astype(int))
        print("Epoch: ", e, "-    Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([test_X, left_trial_padded_seqs, right_trial_padded_seqs], batch_size=1024, verbose=0)
    print('=' * 60)
    return pred_val_y, pred_test_y, best_score

def train_pred(model, train_X, train_y, val_X, val_y, epochs=2, callback=None):
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks = callback, verbose=0)
        
        #model.load_weights(filepath)
        pred_val_y = model.predict(val_X, batch_size=1024, verbose=0)


        # best_score = metrics.f1_score(val_y, (pred_val_y > 0.20).astype(int))
        p_levels, r_levels, best_score, _ = precision_recall_fscore_support(val_y, (pred_val_y > 0.20).astype(int), average="macro")
        print("Epoch: ", e, "-    Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([val_X], batch_size=1024, verbose=0)
    # pred_dev_y = model.predict([dev_X], batch_size=1024, verbose=0)
    pred_trial_y = model.predict([train_X], batch_size=1024, verbose=0)
    print('=' * 60)
    return pred_val_y, pred_test_y, best_score,pred_trial_y
    ############start##################
train_X, train_y, word_index, train_id = load_and_prec()
x_train=train_X
# embedding_matrix_1 = load_glove(word_index)
embedding_matrix_2 = load_fasttext(word_index)
# embedding_matrix_3 = load_para(word_index)

nb_words = min(max_features, len(word_index)+1)

## Simple average: http://aclweb.org/anthology/N18-2031

# We have presented an argument for averaging as
# a valid meta-embedding technique, and found experimental
# performance to be close to, or in some cases 
# better than that of concatenation, with the
# additional benefit of reduced dimensionality  


## Unweighted DME in https://arxiv.org/pdf/1804.07983.pdf

# “The downside of concatenating embeddings and 
#  giving that as input to an RNN encoder, however,
#  is that the network then quickly becomes inefficient
#  as we combine more and more embeddings.”
  
embedding_matrix =  embedding_matrix_2
# embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2), axis = 1)
# embedding_matrix = embedding_matrix_1
np.shape(embedding_matrix)

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

DATA_SPLIT_SEED = 218
clr = CyclicLR(base_lr=0.001, max_lr=0.002,
               step_size=300., mode='exp_range',
               gamma=0.99994)

train_meta = np.zeros(train_y.shape)
# test_meta = np.zeros(test_X.shape[0])
# dev_meta = np.zeros(dev_X.shape[0])
trial_meta = np.zeros(train_X.shape[0])
print(train_X.shape[0])

# MAX_SEQUENCE_LENGTH=maxlen
# # 模型结构：词嵌入*3-LSTM*2-拼接-全连接-最大化池化-全连接
# # 我们需要重新整理数据集
# left_train_word_ids = [[max_features] + x[:-1] for x in train_X_]
# left_test_word_ids = [[max_features] + x[:-1] for x in train_X_]
# right_train_word_ids = [x[1:] + [max_features] for x in train_X_]
# right_test_word_ids = [x[1:] + [max_features] for x in train_X_]

# left_trial_word_ids = [[max_features] + x[:-1] for x in test_X_]
# right_trial_word_ids = [x[1:] + [max_features] for x in test_X_]

# # 分别对左边和右边的词进行编码
# left_train_padded_seqs = pad_sequences(left_train_word_ids, maxlen=MAX_SEQUENCE_LENGTH)
# left_test_padded_seqs = pad_sequences(left_test_word_ids, maxlen=MAX_SEQUENCE_LENGTH)
# right_train_padded_seqs = pad_sequences(right_train_word_ids, maxlen=MAX_SEQUENCE_LENGTH)
# right_test_padded_seqs = pad_sequences(right_test_word_ids, maxlen=MAX_SEQUENCE_LENGTH)

# indices = np.arange(train_X.shape[0])

# left_train_padded_seqs = left_train_padded_seqs[indices]
# left_test_padded_seqs = left_test_padded_seqs[indices]
# right_train_padded_seqs = right_train_padded_seqs[indices]
# right_test_padded_seqs = right_test_padded_seqs[indices]

# left_trial_padded_seqs = pad_sequences(left_trial_word_ids, maxlen=MAX_SEQUENCE_LENGTH)
# right_trial_padded_seqs = pad_sequences(right_train_word_ids, maxlen=MAX_SEQUENCE_LENGTH)

from keras.callbacks import ModelCheckpoint
# filepath="/home/bin_lab/桌面/task5/checkpoint/capsule/weights-improvement--{epoch:02d}--{val_f1:.2f}.hdf5"
# filepath="/home/bin_lab/桌面/task5/checkpoint/capsule/weights.best_capsule_model.hdf5"
# checkpoint=ModelCheckpoint(filepath,monitor='val_f1',verbose=1,save_best_only=True,mode='max')
# callbacks_list=[checkpoint]

splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X, train_y))
# print("splits")
# print(splits)
# print(len(splits))
for idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = train_X[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X[valid_idx]
        y_val = train_y[valid_idx]

        # x_train_left = left_train_padded_seqs[train_idx]
        # x_val_left = left_test_padded_seqs[valid_idx]
        # x_train_right = right_train_padded_seqs[train_idx]
        # x_val_right = right_test_padded_seqs[valid_idx]
        # model = model_lstm_atten(embedding_matrix),0.73
        # model = model_HAN(embedding_matrix)
        # model = model_lstm_HAN(embedding_matrix)
        
        # model = capsulenet_model(embedding_matrix, maxlen, len(word_index)+1, 300, 
        #         1, False)#0.75
        # model = RCNN_Net(embedding_matrix, maxlen, len(word_index)+1, 300, 
        #         1, False)#str(len(data)) + ' arrays: ' + str(data)[:200] + '...')
# ValueError: Error when checking model input: the list of Numpy arrays that you are passing to your model is not the size the model expected. Expected to see 3 
        # model = rnncnn_model(embedding_matrix)#0.75
        # model = kmax_textcnn_model(embedding_matrix)#0.7896
        # model = pooled_gru_model(embedding_matrix)#0.72
        # model = lstm_conv_model(embedding_matrix)#0.76
        # model = gru128_model(embedding_matrix)#0.76
        # model = inceptioncnn_model(embedding_matrix)#0.80.80
        # model = text2dCNN_model(embedding_matrix)   #762
        # model = lstm_att_block_model(embedding_matrix)
        # model = gru_att_block_model(embedding_matrix)
        # model = gru_att_model(embedding_matrix) #10
        # model = model_lstm_atten_1(embedding_matrix)#0.72
        # model = model_gru_srk_atten(embedding_matrix)#0.76
        model = model_lstm_du(embedding_matrix)0.77
        
        # pred_val_y, pred_test_y, best_score = rcnn_train_pred(model, X_train, y_train, X_val, y_val, epochs = 20, callback = [clr,])
        pred_val_y, pred_test_y, best_score,pred_trial_y= train_pred(model, X_train, y_train, X_val, y_val, epochs = 20, callback = [clr,])
        train_meta[valid_idx] = pred_val_y.reshape(-1)


        # test_meta += pred_test_y.reshape(-1) / len(splits)
        # dev_meta += pred_dev_y.reshape(-1) / len(splits)
        # trial_meta += pred_trial_y.reshape(-1) / len(splits)
        # p_levels, r_levels, f1_levels, _ = precision_recall_fscore_support(text_y, pred_test_y, average="macro")
        # print("f1_score:")
        # print(f1_levels)

# print(test_meta)
# # print(pred_test_y)

###############################################################
def save_result(y_pred, file_name,):
    result_df = pd.DataFrame({'ID':test_id, 'is_humor': y_pred})
    result_df.to_csv(file_name, index=False)


save_file = os.path.join('data\\result', 'model-gru_att_block_model.csv')
save_result(train_meta, save_file)


