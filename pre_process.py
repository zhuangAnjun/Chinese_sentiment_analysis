import numpy as np
import pandas as pd
import os
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import backend as K

# 建立词典以及padding成句子
def tok_and_padding( X_train, X_test, max_features=4000, maxlen=64):
    tok = text.Tokenizer(num_words=max_features, lower=False)
    tok.fit_on_texts(X_train + X_test)
    X_train = tok.texts_to_sequences(X_train)
    X_test = tok.texts_to_sequences(X_test)
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='pre')
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='pre')
    return [tok, x_train, x_test]

# 计算词在词典中的位置
def get_embedding_index(EMBEDDING_FILE):
    embeddings_index = {}
    with open(EMBEDDING_FILE, encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# 获得编码矩阵，当词语不存在时，向量为0
def get_embedding_matrix(tok, EMBEDDING_FILE, embed_size=300, max_features=4000):

    word_index = tok.word_index
    num_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    embeddings_index = get_embedding_index(EMBEDDING_FILE)
    for word, i in word_index.items():
        if i >= max_features:
            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
