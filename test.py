import numpy as np 
import pandas as pd 
from keras import initializers, regularizers, constraints, optimizers,callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from random import randint
from pre_process import *
from model import sentiment_analysis
from f1_score import f1

EMBEDDING_FILE_300v = './datasets/word2vec_org'
EMBEDDING_FILE_200v = './datasets/word2vec_org_200v'
EMBEDDING_FILE_400v = './datasets/word2vec_org_400v'

test_x = pd.read_csv('./datasets/test_data.txt',header=None, dtype='str', delimiter='\n')

X_test = np.array(test_x[0][:])
X_test = X_test.tolist()

max_features=4000
maxlen=64



print('Predicting....')
y_pred = model1.predict(x_test,batch_size=1024,verbose=1)
print(y_pred)