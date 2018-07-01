import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from pre_process import *
from model import sentiment_analysis
from f1_score import f1
from sampling import *
from keras.utils import to_categorical

# 预训练好的词向量文件
EMBEDDING_FILE = './model/word2vec_org_500v_cut'

# 读取训练集和测试集文件，记得要先预处理
train_x = pd.read_csv('./datasets/train_data_cut.txt',  header=None, dtype='str', delimiter='\n')
train_y = pd.read_csv('./datasets/train_label', dtype='int', header=None)
test_x = pd.read_csv('./datasets/test_data_cut.txt',header=None, dtype='str', delimiter='\n')

X_test = np.array(test_x[0][:])
X_test = X_test.tolist()
X_train, y_train = sampling(train_x, train_y)

# max_feature为词表大小， maxlen为句子长度
max_features = 4000
maxlen = 64
batch_size = 64
epochs = 3

# tok为词典，返回的数据是经过padding和截取的
tok, x_train, x_test = tok_and_padding( X_train, X_test, max_features)

embedding_matrix_500v = get_embedding_matrix(
    tok, EMBEDDING_FILE, 500, max_features)

# 以0.8的比例划分训练集和验证集,并将label转化成one-hot
W = np.ones((len(X_train),1))
train_data = np.concatenate((np.array(x_train), y_train, W), axis=1)
train, val = train_test_split(train_data, test_size=0.0, random_state=666)
X_tra, y_tra, weights = train[:,0:maxlen], train[:,maxlen:maxlen+1], train[:,maxlen+1:maxlen+2].reshape((-1))
X_val, y_val = val[:,0:maxlen], val[:,maxlen:maxlen+1]
# 将label转化为one-hot
y_tra = to_categorical(y_tra, num_classes=3)
y_val = to_categorical(y_val, num_classes=3)

filepath = "./model/weights_base_best.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_f1', mode="max", patience=5)
callbacks_list = [checkpoint, early]

#5次训练预测取平均值
y_preds = np.zeros((x_test.shape[0],3))
for i in range(10):
    print('The %d times'%(i+1))
    model = sentiment_analysis(embedding_matrix_500v, embed_size=500)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=[f1])
    model.fit(np.array(X_tra), np.array(y_tra), batch_size=256, epochs=5, callbacks = callbacks_list, class_weight = [0.995, 1.1, 0.99],sample_weight=np.array(weights), verbose=1)

    y_preds += model.predict(x_test, batch_size=1024, verbose=1)
y_preds = y_preds/10.0

#半监督过程，weight表示样本权值
preds = y_preds
for i in range(9):
    test_data = sampling_from_pred(x_test, preds/(i+1), weight=20.0-i, threshold=[0.998+i/10000, 0.8+i/5, 0.999995+i/10000000])
    print(test_data.shape[0])
    #将采样出来的测试数据与训练数据拼接
    Train_data = np.concatenate((train_data, test_data), axis=0)
    #随机化
    np.random.shuffle(Train_data)
    #将x，y，weight分离出来
    X_tra = Train_data[:,0:maxlen]
    y_tra = Train_data[:,maxlen:maxlen+1]
    #weight要reshape成1D形状
    weights = Train_data[:,maxlen+1:maxlen+2].reshape((-1))
    #将y转化成one-hot编码
    y_tra = to_categorical(y_tra, num_classes=3)
    #重新建图
    model = sentiment_analysis(embedding_matrix_500v, embed_size=500)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=[f1])
    #在fit的时候同时对类别和样本加权
    model.fit(np.array(X_tra), np.array(y_tra), batch_size=256, epochs=5, callbacks = callbacks_list, class_weight = [0.995, 1.1, 0.99],sample_weight=np.array(weights), verbose=1)

    #将预测值叠加，然后取平均，能避免某次出现很坏的情况
    preds += model.predict(x_test, batch_size=1024, verbose=1)
preds = preds/9.0

# 输出预测值的情况
y_pred = np.argmax(preds, axis=1)
print(y_pred.shape)
num = [0, 0, 0]
for i in range(y_pred.shape[0]):
    num[y_pred[i]] += 1
print(num)

# 将得到的预测值写入文件
result = open("./datasets/submission.csv", "w", encoding='UTF-8')
for i in range(y_pred.shape[0]):
    result.write("{},{}\n".format(i, y_pred[i]))
result.close()
