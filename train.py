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

# 预训练好的词向量文件
EMBEDDING_FILE_300v = './datasets/word2vec_org'
EMBEDDING_FILE_200v = './datasets/word2vec_org_200v'
EMBEDDING_FILE_400v = './datasets/word2vec_org_400v'

# 读取训练集和测试集文件，记得要先预处理
train_x = pd.read_csv('./datasets/train_data.txt',
                      header=None, dtype='str', delimiter='\n')
train_y = pd.read_csv('./datasets/train_label', dtype='int', header=None)
test_x = pd.read_csv('./datasets/test_data.txt',
                     header=None, dtype='str', delimiter='\n')

X_test = np.array(test_x[0][:])
X_test = X_test.tolist()
X_train, y_train = sampling(train_x, train_y)

# max_feature为词表大小， maxlen为句子长度
max_features = 4000
maxlen = 64
batch_size = 64
epochs = 10

# tok为词典，返回的数据是经过padding和截取的
tok, x_train, x_test = tok_and_padding( X_train, X_test, max_features)

# 得到不同维度的编码矩阵
embedding_matrix_200v = get_embedding_matrix(
     tok, EMBEDDING_FILE_200v,200,max_features)
embedding_matrix_300v = get_embedding_matrix(
    tok, EMBEDDING_FILE_300v, 300, max_features)
embedding_matrix_400v = get_embedding_matrix(
    tok, EMBEDDING_FILE_400v, 400, max_features)

model = sentiment_analysis(embedding_matrix_200v, embedding_matrix_300v,
                           embedding_matrix_400v, maxlen, [True, True, True],max_features)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3), metrics=[f1])

# 以0.8的比例划分训练集和验证集
X_tra, X_val, y_tra, y_val = train_test_split(
    x_train, y_train, train_size=0.8, random_state=666)

filepath = "./model/weights_base_best.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_f1', mode="max", patience=5)
callbacks_list = [checkpoint, early]

model.fit(np.array(X_tra), np.array(y_tra), batch_size=batch_size, epochs=epochs,
           validation_data=(np.array(X_val), np.array(y_val)), callbacks=callbacks_list, verbose=1)
print('complete')

# 载入最佳权重
model.load_weights('./model/weights_base_best.hdf5')
print('Predicting....')
y_pred = model.predict(x_test, batch_size=1024, verbose=1)

# 输出预测值的情况
y_pred = np.argmax(y_pred, axis=1)
print(y_pred.shape)
num = [0, 0, 0]
for i in range(y_pred.shape):
    num[y_pred[i]] += 1
print(num)

# 将得到的预测值写入文件
result = open("./datasets/submission.csv", "w", encoding='UTF-8')
for i in range(y_pred.shape[0]):
    result.write("{},{}\n".format(i, y_pred[i]))
result.close()
