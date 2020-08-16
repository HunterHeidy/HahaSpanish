
# #################### Load packages and data #########################
# import os
# print(os.listdir("../input"))

## some config values 
embed_size = 300 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 50 # max number of words in a question to use

import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
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

from keras.initializers import *
from keras.layers import *
from keras.models import *
from sklearn.model_selection import train_test_split, StratifiedKFold
# from capsule_layer import CategoryCap, PrimaryCap, Length, Mask

from keras_wc_embd import get_embedding_layer
from keras_wc_embd import get_dicts_generator
from keras_wc_embd import get_embedding_layer, get_embedding_weights_from_file

import tensorflow as tf

from keras_targeted_dropout import TargetedDropout

from sklearn.model_selection import train_test_split

from keras.utils.vis_utils import plot_model

def load_and_prec():
    train_df = pd.read_csv("/home/bin_lab/桌面/task6/data/cleaned_train_data.csv")
    test_df = pd.read_csv("/home/bin_lab/桌面/task6/data/cleaned_trial_data.csv")
    dev_df = pd.read_csv("/home/bin_lab/桌面/task6/data/cleaned_data_testset-taska.csv")
    trial_df = pd.read_csv("/home/bin_lab/桌面/task6/data/cleaned_trial_data.csv")
    # train_df = pd.read_csv("/home/bin_lab/桌面/task6/cleaned_train_data.csv")
    # test_df = pd.read_csv("/home/bin_lab/桌面/task6/cleaned_trial_data.csv")


    # train_df = pd.read_csv("/home/bin_lab/桌面/task9A/data/Subtask-A-master/cleaned_train_data_9.csv")
    # test_df = pd.read_csv("/home/bin_lab/桌面/task9A/data/Subtask-A-master/SubtaskA_Trial_Test.csv")

    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    print("Dev shape : ",dev_df.shape)
    print("Trial shape : ",trial_df.shape)
    
    ## fill up the missing values
    train_X = train_df["comment_text"].fillna("_##_").values
    test_X = test_df["comment_text"].fillna("_##_").values#
    dev_X = dev_df["comment_text"].fillna("_##_").values
    trial_X = trial_df["comment_text"].fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X_ = tokenizer.texts_to_sequences(train_X)
    test_X_ = tokenizer.texts_to_sequences(test_X)
    dev_X_ = tokenizer.texts_to_sequences(dev_X)
    trial_X_ = tokenizer.texts_to_sequences(trial_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X_, maxlen=maxlen)
    test_X = pad_sequences(test_X_, maxlen=maxlen)
    dev_X = pad_sequences(dev_X_, maxlen=maxlen)
    trial_X = pad_sequences(trial_X_, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['label1'].values
    test_y = test_df['label1'].values
    # train_y = train_df['label1'].values
    # test_y = test_df['label1'].values
    
    #shuffling the data
    np.random.seed(218)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]

    train_id = train_df['id']
    # test_id = test_df['id']
    dev_id = dev_df['id']
    # trial_id = trial_df['id']

    sequence_train = train_df["comment_text"].fillna("_##_").values.astype(str).tolist()
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
    
    
    return train_X, test_X, train_y, tokenizer.word_index, test_y, train_X_, test_X_, dev_X, dev_id, trial_X, train_id

############################ loading embedding #################################
def load_glove(word_index):
    EMBEDDING_FILE = '/home/bin_lab/桌面/n2c2-1/data/glove/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

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
    
def load_fasttext(word_index):    
    # EMBEDDING_FILE = '/home/bin_lab/桌面/n2c2-1/data/crawl-300d-2M.vec'
    # def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    # embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    EMBEDDING_FILE = '/home/bin_lab/桌面/n2c2-1/data/crawl-300d-2M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    #embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='gb18030'))

    # EMBEDDING_FILE = '/media/bin_lab/C4F6073207B3A949/Linux/data/glove.840B.300d.txt'
    # def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE) if len(o)>100)

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

def load_para(word_index):
    EMBEDDING_FILE = '/home/bin_lab/桌面/n2c2-1/data/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

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

######################## Capsule layer ###########################
def squash(x, axis=-1):
    # # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = b + K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

######################## Attention layer ##########################
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

class HAN_AttLayer(Layer):
    def __init__(self, init='glorot_uniform', kernel_regularizer=None, 
                 bias_regularizer=None, kernel_constraint=None, 
                 bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get(init)
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(kernel_regularizer)
        
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        super(HAN_AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        
        self.built = True
        
    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W) # (x, 40, 1)
        uit = K.squeeze(uit, -1) # (x, 40)
        uit = uit + self.b # (x, 40) + (40,)
        uit = K.tanh(uit) # (x, 40)

        ait = uit * self.u # (x, 40) * (40, 1) => (x, 1)
        ait = K.exp(ait) # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            ait = mask*ait #(x, 40) * (x, 40, )

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]

        # return flattened output
        return Flatten()(top_k)

###################### F1 score and CLR #######################
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
    

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

################## Building model #####################
def model_lstm_atten(embedding_matrix):
    
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
    
    atten_1 = Attention(maxlen)(x) # skip connect
    atten_2 = Attention(maxlen)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    # conc = Dropout(0.5)(conc)
    conc = TargetedDropout(drop_rate=0.5, target_rate=0.5)(conc)
    # conc = BatchNormalization()(conc)

    outp = Dense(1, activation="sigmoid")(conc)    

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    
    return model

def model_HAN(embedding_matrix):
    
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
    
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
    x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)   #, kernel_initializer=glorot_normal(seed=1230), recurrent_initializer=orthogonal(gain=1.0, seed=1000)

    # x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
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
    forward = CuDNNGRU(300, return_sequences = True)(l_embedding) # 等式(1)
    # 等式(2)
    backward = CuDNNGRU(300, return_sequences = True, go_backwards = True)(r_embedding) 
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
    plot_model(model, to_file="model.png", show_shapes=True)

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
    
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    x = attention_3d_block(x)
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
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
    
    x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
    x = attention_3d_block(x)
    x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
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
    
    x0 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x1 = attention_3d_block(x0)
    x2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(x1)
    x3 = Add()([x0, x2])
    x4 = Bidirectional(CuDNNGRU(64, return_sequences=True))(x3)
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
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def model_gru_srk_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(nb_words, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
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
    x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
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

        # best_score = metrics.f1_score(val_y, (pred_val_y > 0.33).astype(int))
        p_levels, r_levels, best_score, _ = precision_recall_fscore_support(val_y, (pred_val_y > 0.33).astype(int), average="macro")
        print("Epoch: ", e, "-    Val F1 Score: {:.4f}".format(best_score))

    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    pred_dev_y = model.predict([dev_X], batch_size=1024, verbose=0)
    pred_trial_y = model.predict([x_train], batch_size=1024, verbose=0)
    print('=' * 60)
    return pred_val_y, pred_test_y, best_score, pred_dev_y, pred_trial_y


########### Main part: load, train, pred and blend #############
train_X, test_X, train_y, word_index, test_y, train_X_, test_X_, dev_X, dev_id, trial_X, train_id = load_and_prec()
x_train=train_X
embedding_matrix_1 = load_glove(word_index)
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
  
embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2], axis = 0)
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
test_meta = np.zeros(test_X.shape[0])
dev_meta = np.zeros(dev_X.shape[0])
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
print("splits")
print(splits)
print(len(splits))
for idx, (train_idx, valid_idx) in enumerate(splits):
        X_train = train_X[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X[valid_idx]
        y_val = train_y[valid_idx]

        # x_train_left = left_train_padded_seqs[train_idx]
        # x_val_left = left_test_padded_seqs[valid_idx]
        # x_train_right = right_train_padded_seqs[train_idx]
        # x_val_right = right_test_padded_seqs[valid_idx]
        # model = model_lstm_atten(embedding_matrix)
        # model = model_HAN(embedding_matrix)
        # model = model_lstm_HAN(embedding_matrix)
        
        # model = capsulenet_model(embedding_matrix, maxlen, len(word_index)+1, 300, 
        #         1, False)
        # model = RCNN_Net(embedding_matrix, maxlen, len(word_index)+1, 300, 
        #         1, False)
        # model = rnncnn_model(embedding_matrix)
        model = kmax_textcnn_model(embedding_matrix)
        # model = pooled_gru_model(embedding_matrix)
        # model = lstm_conv_model(embedding_matrix)
        # model = gru128_model(embedding_matrix)
        # model = inceptioncnn_model(embedding_matrix)
        # model = text2dCNN_model(embedding_matrix)   #10
        # model = lstm_att_block_model(embedding_matrix)
        #model = gru_att_block_model(embedding_matrix)
        # model = gru_att_model(embedding_matrix) #10
        # model = model_lstm_atten_1(embedding_matrix)
        # model = model_gru_srk_atten(embedding_matrix)
        # model = model_lstm_du(embedding_matrix)
        # pred_val_y, pred_test_y, best_score = rcnn_train_pred(model, X_train, y_train, X_val, y_val, epochs = 20, callback = [clr,])
        pred_val_y, pred_test_y, best_score, pred_dev_y, pred_trial_y = train_pred(model, X_train, y_train, X_val, y_val, epochs = 20, callback = [clr,])
        train_meta[valid_idx] = pred_val_y.reshape(-1)
        test_meta += pred_test_y.reshape(-1) / len(splits)
        dev_meta += pred_dev_y.reshape(-1) / len(splits)
        trial_meta += pred_trial_y.reshape(-1) / len(splits)
        # p_levels, r_levels, f1_levels, _ = precision_recall_fscore_support(text_y, pred_test_y, average="macro")
        # print("f1_score:")
        # print(f1_levels)

# print(test_meta)
# # print(pred_test_y)
###############################################################
def save_result(y_pred, file_name, data_id):
    result_df = pd.DataFrame({'ID':data_id, 'label2': y_pred})
    result_df.to_csv(file_name, index=False)


save_file = os.path.join('/home/bin_lab/桌面/task6/result/gailv/dev', 'b_dev_gailv_clr_kmax_cnn-model5.csv')
save_result(test_meta, file_name=save_file, data_id=None)

# print(test_meta)
print(test_meta.shape)
np.save('/home/bin_lab/桌面/task6/result/gailv/dev/b_dev_clr_kmax_cnn-model5.npy', test_meta)

#### test ####
save_file = os.path.join('/home/bin_lab/桌面/task6/result/gailv/test', 'b_test_final_gailv_clr_kmax_cnn-model5.csv')
save_result(dev_meta, file_name=save_file, data_id=dev_id)

# print(test_meta)
print(dev_meta.shape)
np.save('/home/bin_lab/桌面/task6/result/gailv/test/b_test_final_clr_kmax_cnn-model5.npy', dev_meta)
#### trial ####
save_file = os.path.join('/home/bin_lab/桌面/task6/result/gailv/train', 'b_train_final_gailv_clr_kmax_cnn-model5.csv')
save_result(trial_meta, file_name=save_file, data_id=train_id)

# print(test_meta)
print(trial_meta.shape)
np.save('/home/bin_lab/桌面/task6/result/gailv/train/b_train_final_clr_kmax_cnn-model5.npy', trial_meta)
#################################################################3
sub = pd.read_csv('/home/bin_lab/桌面/task6/data/cleaned_train_data_b_f.csv')
sub.prediction = test_meta > 0.33
sub.to_csv("submission_trial.csv", index=False)

print(f1_score(y_true=train_y, y_pred=train_meta > 0.33))

sub_1 = pd.read_csv('/home/bin_lab/桌面/task6/data/cleaned_dev_data_b_f.csv')
sub_1.prediction = test_meta > 0.33
sub_1.to_csv("submission_dev.csv", index=False)

print(f1_score(y_true=train_y, y_pred=train_meta > 0.33))


#model.load_weights(filepath)
#y_pred = model.predict(cnn_data_test)

# y_pred = np.argmax(model.predict(test_X), axis=-1)
# print(y_pred)



def tranfer(array_y):
    array_y[array_y > 0.5]=1
    array_y[array_y <= 0.5]=0

    return array_y.astype(int)

test_meta_y=tranfer(test_meta)
# print(test_meta_y)
dev_meta_y=tranfer(dev_meta)
trial_meta_y=tranfer(trial_meta)

p_levels, r_levels, f1_levels, _ = precision_recall_fscore_support(test_y, test_meta_y, average="macro")

print("Macro:")
print(f1_levels)

# print("test_meta:")
# print(test_meta.shape)

# print("pred_test_y:")
# print(pred_test_y.shape)

#################################################################
save_file = os.path.join('/home/bin_lab/桌面/task6/result/label/dev', 'b_dev_clr_kmax_cnn-model5.csv')
save_result(test_meta_y, file_name=save_file, data_id=None)

# print(test_meta)
# print(test_meta.shape)
np.save('/home/bin_lab/桌面/task6/result/label/dev/b_dev_clr_kmax_cnn-model5_1.npy', test_meta)

#### test ####
save_file = os.path.join('/home/bin_lab/桌面/task6/result/label/test', 'b_test_final_clr_kmax_cnn-model5.csv')
save_result(dev_meta_y, file_name=save_file, data_id=dev_id)

# print(test_meta)
# print(test_meta.shape)
np.save('/home/bin_lab/桌面/task6/result/label/test/b_test_final_lr_kmax_cnn-model5_1.npy', dev_meta)
#### trial ####
save_file = os.path.join('/home/bin_lab/桌面/task6/result/label/train', 'b_train_final_clr_kmax_cnn-model5.csv')
save_result(trial_meta_y, file_name=save_file, data_id=None)

# print(test_meta)
# print(test_meta.shape)
np.save('/home/bin_lab/桌面/task6/result/label/train/b_train_final_lr_kmax_cnn-model5_1.npy', trial_meta)
##################################################################