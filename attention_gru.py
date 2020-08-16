from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Dense,Embedding,Activation,merge,Input,Lambda,Reshape,BatchNormalization
from keras.layers import Convolution1D,Flatten,Dropout,MaxPool1D,GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D
from keras.layers import Conv1D,Bidirectional,LSTM,GRU, CuDNNLSTM, CuDNNGRU, concatenate
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
import gensim
import pandas as pd
import numpy as np
import os
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
from keras.utils import to_categorical

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from keras_targeted_dropout import TargetedDropout

#clean_questions=pd.read_csv("data/train_clean_data.csv")
clean_questions=pd.read_csv("data/cleaned_data_train_haha.csv")

# trial_data=pd.read_csv("/home/bin_lab/桌面/WASSA/cleaned_data/cleaned_dev_data.csv")

#test_data=pd.read_csv("data/trial_clean_data.csv")
# test_data=pd.read_csv("/home/bin_lab/桌面/WASSA/cleaned_data/cleaned_test_data.csv")

from nltk.tokenize import RegexpTokenizer

tokenizer=RegexpTokenizer(r'\w+')
#正则表达式分词器
clean_questions["tokens"]=clean_questions["text"].astype(str).apply(tokenizer.tokenize)
print(clean_questions.head())

# trial_data["tokens"]=clean_questions["comment_text"].astype(str).apply(tokenizer.tokenize)

# test_data["tokens"]=clean_questions["comment_text"].astype(str).apply(tokenizer.tokenize)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

all_words=[word for tokens in clean_questions["tokens"] for word in tokens]
sentence_lengths=[len(tokens) for tokens in clean_questions["tokens"]]
VOCAB=sorted(list(set(all_words)))
print("%s words total,with a vocabulary size of %s" % (len(all_words),len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))

# word2vec_path = "/media/bin_lab/C4F6073207B3A949/Linux/python/kaggle/Toxic_Comment_Classification_Challenge/GoogleNews-vectors-negative300.bin"
# word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = max(sentence_lengths)+1
VOCAB_SIZE = len(VOCAB)
NUM_CLASSES=2

label2emotion={0:'humor',1:'NotHumor'}
emotion2label={'humor':0,'NotHumor':1}

# label2emotion = {0:"sad", 1:"disgust", 2: "fear", 3:"angry", 4:"surprise", 5:"joy"}
# emotion2label = {"sad":0, "disgust":1, "fear":2, "angry":3, "surprise":4, "joy":5}

VALIDATION_SPLIT=.2
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(clean_questions["text"].astype(str).tolist())
# tokenizer.fit_on_texts(trial_data["comment_text"].astype(str).tolist())
# tokenizer.fit_on_texts(test_data["comment_text"].astype(str).tolist())
sequences_train = tokenizer.texts_to_sequences(clean_questions["text"].astype(str).tolist())#转换成列表
# sequences_trial = tokenizer.texts_to_sequences(trial_data["comment_text"].astype(str).tolist())
# sequences_test = tokenizer.texts_to_sequences(test_data["comment_text"].astype(str).tolist())

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))#训练样例总数

cnn_data = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)#keras只能接受长度相同的序列输入。
# 因此如果目前序列长度参差不齐，这时需要使用pad_sequences()。
# 该函数是将序列转化为经过填充以后的一个长度相同的新序列新序列。
# cnn_data_trial = pad_sequences(sequences_trial, maxlen=MAX_SEQUENCE_LENGTH)
# cnn_data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(clean_questions["votes"]))#热编码转换

indices = np.arange(cnn_data.shape[0])
np.random.shuffle(indices)
# 函数shuffle与permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序）；
# 区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
# 而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
# --------------------- 

cnn_data = cnn_data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])

# embedding_weights = np.zeros((len(word_index)+1, EMBEDDING_DIM))
# for word,index in word_index.items():
#     embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
# print(embedding_weights.shape)

def get_embeddings_FastText():
    from gensim.models.keyedvectors import KeyedVectors
    w2v_bin = 'data/fasttext-spanish/cc.es.300.vec'
    model = KeyedVectors.load_word2vec_format(w2v_bin, binary=False)

    embedding_weights = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = model[word] if word in model else None
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_weights[i] = embedding_vector

    return embedding_weights

embedding_weights = get_embeddings_FastText()
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

def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 6
    macroRecall /= 6
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[:].sum()
    falsePositives = falsePositives[:].sum()
    falseNegatives = falseNegatives[:].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def model_lstm_atten(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    
    inp = Input(shape=(max_sequence_length,))
    x = Embedding(num_words, embedding_dim, weights=[embeddings], trainable=False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(150, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(150, return_sequences=True))(x)
    
    atten_1 = Attention(max_sequence_length)(x) # skip connect
    atten_2 = Attention(max_sequence_length)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(256, activation="relu")(conc)
    # conc = Dropout(0.5)(conc)
    conc = TargetedDropout(drop_rate=0.5, target_rate=0.5)(conc)
    outp = Dense(labels_index, activation="softmax")(conc)    

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model

# x_train = cnn_data[:-num_validation_samples]
# y_train = labels[:-num_validation_samples]
# x_val = cnn_data[-num_validation_samples:]
# y_val = labels[-num_validation_samples:]


from keras.callbacks import ModelCheckpoint
filepath="/home/bin_lab/桌面/WASSA/checkpoint/weights.best_v3_atten_lstm.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callbacks_list=[checkpoint]

print("######## Starting k-fold cross validation ########")

NUM_FOLDS=5

metrics = {"accuracy" : [],
               "microPrecision" : [],
               "microRecall" : [],
               "microF1" : []}

val_meta=[]
test_meta=[]
trial_meta=[]

model = model_lstm_atten(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index)+1, EMBEDDING_DIM, 
                    len(list(clean_questions["label"].unique())))

model.summary()

for k in range(NUM_FOLDS):
    print('-'*40)
    print("Fold %d/%d" % (k+1, NUM_FOLDS))
    validationSize = int(len(cnn_data)/NUM_FOLDS)
    index1 = validationSize * k
    index2 = validationSize * (k+1)

    x_train = np.vstack((cnn_data[:index1], cnn_data[index2:]))
    y_train = np.vstack((labels[:index1], labels[index2:]))
    x_val = cnn_data[index1:index2]
    y_val = labels[index1:index2]

#drs=np.array([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6])
# rdrs=np.array([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6])
# epoch=np.array([20,30,40,50])
# batchs=np.array([64,128,256,512,1024,2048])
# kernel_size1=np.array([100,200,256,300,350,400])
# kernel_size2=np.array([100,200,256,300,350,400])

    

# model=KerasClassifier(build_fn=model,verbose=0)

# param_grid=dict(dr=drs)#,rdr=rdrs,epochs=epoch,batch_size=batchs,kernel1=kernel_size1,kernel2=kernel_size2)

# grid=GridSearchCV(estimator=model,param_grid=param_grid)
# grid_result=grid.fit(x_train,y_train)

# print("Best: %f using %s" % (grid_result.best_score_,grid_result.best_parms_))
# for params,mean_score,scores in grid_result.grid_scores_:
#   print("%f (%f) with: %r" % (scores.mean(),scores.std(),params))



    model.fit(x_train, y_train,
              batch_size=512,
              epochs=8,
              validation_data=(x_val, y_val), verbose=2, callbacks=None)

    predictions = model.predict(x_val, batch_size=256)
    val_meta.append(predictions)
    print(predictions)
    predictions_test = model.predict(cnn_data_test)
    predictions_trial = model.predict(cnn_data_trial)

    test_meta.append(predictions_test)
    trial_meta.append(predictions_trial)
    # p_levels, r_levels, f1_levels, _ = precision_recall_fscore_support(y_val, predictions, average="macro")
    accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, y_val)
    metrics["accuracy"].append(accuracy)
    metrics["microPrecision"].append(microPrecision)
    metrics["microRecall"].append(microRecall)
    metrics["microF1"].append(microF1)

print("\n################## Metrics ###################")
print("Average Micro Precision : %.4f" % (sum(metrics["microPrecision"])/len(metrics["microPrecision"])))
print("Average Micro Recall : %.4f" % (sum(metrics["microRecall"])/len(metrics["microRecall"])))
print("Average Micro F1 : %.4f" % (sum(metrics["microF1"])/len(metrics["microF1"])))

scores=model.evaluate(x_val,y_val,verbose=0)
print("Acuracy: %.2f%%" % (scores[1]*100))

def save_result(y_pred, file_name):
    result_df = pd.DataFrame({'Label': y_pred})
    result_df.to_csv(file_name, index=False)

# model.load_weights(filepath)
#y_pred = model.predict(cnn_data_test)

y_pred = np.argmax(model.predict(cnn_data_test), axis=-1)
print(y_pred)
save_file = os.path.join('/home/bin_lab/桌面/WASSA/result', 'test_atten_gru_model.csv')
save_result(y_pred, file_name=save_file)

y_pred_trial = np.argmax(model.predict(cnn_data_trial), axis=-1)
print(y_pred_trial)
save_file = os.path.join('/home/bin_lab/桌面/WASSA/result', 'trial_atten_gru_model.csv')
save_result(y_pred_trial, file_name=save_file)

y_pred=np.argmax(sum(test_meta)/NUM_FOLDS, axis=-1)
print(y_pred)
save_file = os.path.join('/home/bin_lab/桌面/WASSA/result', 'fold_test_atten_gru_model.csv')
save_result(y_pred, file_name=save_file)

y_pred=np.argmax(sum(trial_meta)/NUM_FOLDS, axis=-1)
print(y_pred)
save_file = os.path.join('/home/bin_lab/桌面/WASSA/result', 'fold_trial_atten_gru_model.csv')
save_result(y_pred_trial, file_name=save_file)