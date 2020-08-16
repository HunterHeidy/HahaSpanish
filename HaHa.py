## some config values 
embed_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from tqdm import tqdm
mol/=12reimport math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
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

import tensorflow as tf

from keras_targeted_dropout import TargetedDropout

from sklearn.model_selection import train_test_split

from keras.utils.vis_utils import plot_model

import keras.preprocessing.text as T
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier 
from Modelpre import CyclicLR,Attention,HAN_AttLayer,Capsule, KMaxPooling,Fasttext





def load_and_prec():


	train_df = pd.read_csv("data/cleaned_data_train_haha.csv")
	test_df = pd.read_csv("data/cleaned_data_test_haha.csv")   
    ## fill up the missing values
    
	train_X = train_df["text"].fillna("_##_").values
	test_X = test_df["text"].fillna("_##_").values
    # dev_X = dev_df["comment_text"].fillna("_##_").values
    # trial_X = trial_df["comment_text"].fillna("_##_").values

	## Tokenize the sentences
	tokenizer = Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(list(train_X))
	train_X_ = tokenizer.texts_to_sequences(train_X)
	test_X_ = tokenizer.texts_to_sequences(test_X)
    # dev_X_ = tokenizer.texts_to_sequences(dev_X)
    # trial_X_ = tokenizer.texts_to_sequences(trial_X)

    ## Pad the sentences 
	train_X = pad_sequences(train_X_, maxlen=maxlen)
	test_X = pad_sequences(test_X_, maxlen=maxlen)
    # dev_X = pad_sequences(dev_X_, maxlen=maxlen)
    # trial_X = pad_sequences(trial_X_, maxlen=maxlen)

    ## Get the target values
	train_y = train_df['is_humor'].values
	# test_y = test_df['is_humor'].values
    # train_y = train_df['label1'].values
    # test_y = test_df['label1'].values
    
    #shuffling the data
	np.random.seed(218)
	trn_idx = np.random.permutation(len(train_X))

	train_X = train_X[trn_idx]
	train_y = train_y[trn_idx]

	train_id = train_df['id']
	test_id = test_df['id']
    # dev_id = dev_df['id']
    # trial_id = trial_df['id']

	sequence_train = train_df["text"].fillna("_##_").values.astype(str).tolist()
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
    
    
	return train_X, test_X, train_y, tokenizer.word_index,test_id, train_id,test_X_,train_X_

filter_sizes = [1,2,3,6]
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
    conv_3 = Conv1D(num_filters, kernel_size=(2),
                                 kernel_initializer='he_normal', activation='elu')(conv_3)
    
    maxpool_0 = MaxPool1D(pool_size=(maxlen - filter_sizes[0] + 1))(conv_0)
    maxpool_1 = MaxPool1D(pool_size=(maxlen - filter_sizes[1] + 1))(conv_1)
    maxpool_2 = MaxPool1D(pool_size=(maxlen - filter_sizes[2] + 1))(conv_2)
    maxpool_3 = MaxPool1D(pool_size=(maxlen - 8 + 1))(conv_3)
        
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = BatchNormalization()(z)
        
    outp = Dense(1, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[f1])

    return model
def train_pred(model, train_X, train_y, val_X, val_y, epochs=2, callback=None):
    for e in range(epochs):
        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y), callbacks = callback, verbose=0)
        
        #model.load_weights(filepath)
        pred_val_y = model.predict(val_X, batch_size=1024, verbose=0)

        # best_score = metrics.f1_score(val_y, (pred_val_y > 0.33).astype(int))
        p_levels, r_levels, best_score, _ = precision_recall_fscore_support(val_y, (pred_val_y > 0.20).astype(int), average="macro")
        print("Epoch: ", e, "-    Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    # pred_dev_y = model.predict([dev_X], batch_size=1024, verbose=0)
    # pred_trial_y = model.predict([x_train], batch_size=1024, verbose=0)
    print('=' * 60)
    return pred_val_y, pred_test_y, best_score
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





################################# main #################################################################
train_X, test_X, train_y, word_index,test_id, train_id,test_X_,train_X_ = load_and_prec()
x_train=train_X
embedding_matrix_1 = Fasttext.load_Glove_SBWC(word_index)#0.79
# embedding_matrix_2 = Fasttext.load_fasttext_wekipadia(word_index)#0.78
# embedding_matrix_3 = Fasttext.load_fasttext_SBWC(word_index)#0.79
embedding_matrix_4 = Fasttext.load_fasttext_cc(word_index)

nb_words = min(max_features, len(word_index)+1)

embedding_matrix = np.mean([embedding_matrix_1,embedding_matrix_4], axis = 0)

# embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_4), axis = 1)

np.shape(embedding_matrix_1)

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
test_meta = np.zeros(test_X.shape[0])
# dev_meta = np.zeros(dev_X.shape[0])
trial_meta = np.zeros(train_X.shape[0])
print(train_X.shape[0])



from keras.callbacks import ModelCheckpoint


splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X, train_y))
# print("splits")
# print(splits)
# print(len(splits))
for idx, (train_idx, valid_idx) in enumerate(splits):
    

	X_train = train_X[train_idx]
	y_train = train_y[train_idx]
	X_val = train_X[valid_idx]
	y_val = train_y[valid_idx]
    
model_1 = inceptioncnn_model(embedding_matrix)#0.7896

model_2 = inceptioncnn_model(embedding_matrix)#0.80.80
lr = LogisticRegression() 
model = StackingClassifier(classifiers=[model_1, model_2], use_probas=True, average_probas=False, meta_classifier=lr)

pred_val_y, pred_test_y, best_score= train_pred(model, X_train, y_train, X_val, y_val, epochs = 20, callback = [clr,])
train_meta[valid_idx] = pred_val_y.reshape(-1)
#saver
checkpointer = ModelCheckpoint(filepath="data/result/checkpoint/inceptioncnn_model.hdf5", verbose=1, save_best_only=True)

###############################################################
pred=model.predict(test_X)

def save_result(y_pred, file_name,test_id):
    result_df = pd.DataFrame({'ID':test_id, 'is_humor': y_pred})
    result_df.to_csv(file_name)

def tranfer(array_y):
    array_y[array_y > 0.5]=1
    array_y[array_y <= 0.5]=0

    return array_y.astype(int)

y_pred=tranfer(pred)
# print(test_meta_y)
# dev_meta_y=tranfer(dev_meta)
# trial_meta_y=tranfer(trial_meta)

# p_levels, r_levels, f1_levels, _ = precision_recall_fscore_support(test_y, test_meta_y, average="macro")

save_file = os.path.join('data\\result', 'model-text2dCNN_model.csv')
save_result(y_pred, save_file,test_id)