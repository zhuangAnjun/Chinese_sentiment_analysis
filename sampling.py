import numpy as np
from random import randint
from keras.utils import to_categorical
import numpy as np
# 对数据进行欠采样和过采样，使数据平衡。 注意输入为原始读取文件获得的数据
def sampling(train_x, train_y, numOfdatas=25000, isSampling=False):
    # 进行采样
    if isSampling:
        X_train = []
        y_train = []

        index = 0
        for i in range(numOfdatas):
            for j in range(0, 3):
                while train_y[0][index] != j:
                    index = randint(0, 82000)
                X_train.append(train_x[0][index])
                y_train.append(train_y[0][index])

        y_train = to_categorical(y_train, num_classes=3)

        return X_train, y_train
        
    # 不进行采样
    else:
        X_train = np.array(train_x[0:82000][0])
        X_train = X_train.tolist()
        y_train = to_categorical(train_y, num_classes=3)
        y_train = np.array(y_train[0:82000][:])
        y_train = y_train.tolist()
    return X_train, y_train


    
#从test数据集中根据阈值进行采样，返回x,y,weight。weight默认得分除以10
def sampling_from_pred(test_data, preds, weight=10.0, threshold=[0.998, 0.98, 0.9999]):
    X = []
    y = []
    W = []
    for n in range(len(preds)):
        index = np.argmax(preds[n])
        if preds[n][index]>=threshold[index]:
            X.append(test_data[n])
            y.append(index)
            W.append(preds[n][index]/weight)
    print(len(y))
    return np.concatenate((np.array(X), np.array(y).reshape(-1,1), np.array(W).reshape(-1,1)), axis=1)