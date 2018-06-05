from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import backend as K

# embedding层，有两个分支，可选择是否可训练
def embedding_block(sequence_input, embed_size, embedding_matrix, trainable1=False, trainable2=True, max_features=4000):
    x1 = Embedding(max_features, embed_size, weights=[
                   embedding_matrix], trainable=trainable1)(sequence_input)
    x2 = Embedding(max_features, embed_size, weights=[
                   embedding_matrix], trainable=trainable2)(sequence_input)

    return [x1, x2]

# 卷积层，返回第二层卷积和第三层卷积结果, 激活函数为relu
def conv1d_block(input1):
    x = Conv1D(64, kernel_size=5, activation='relu', padding="valid",
               kernel_initializer="glorot_uniform")(input1)
    x_1 = Conv1D(64, kernel_size=3, activation='relu',
                 padding="valid", kernel_initializer="glorot_uniform")(x)
    x_2 = Conv1D(128, kernel_size=3, activation='relu',
                 padding="valid", kernel_initializer="glorot_uniform")(x_1)

    return [x_1, x_2]

# 池化层，将卷积的结果进行平均池化和最大池化,再concatenate.
# 输入为list
def pooling_block(x):
    x_1, x_2 = x
    x_avg_1 = GlobalAveragePooling1D()(x_1)
    x_max_1 = GlobalMaxPooling1D()(x_1)
    x_avg_2 = GlobalAveragePooling1D()(x_2)
    x_max_2 = GlobalMaxPooling1D()(x_2)
    return concatenate([x_avg_1, x_max_1, x_avg_2, x_max_2])


def sentiment_analysis(embedding_matrix_200v, embedding_matrix_300v, embedding_matrix_400v, maxlen=64, which=[False, True, False], max_features=4000):
    sequence_input = Input(shape=(maxlen, ))

    block = []

    if which[0]:
        x_200v_1, x_200v_2 = embedding_block(
            sequence_input, 200, embedding_matrix_200v, trainable1=False, trainable2=True, max_features=max_features)
        x_200v_1 = conv1d_block(x_200v_1)
        x_200v_2 = conv1d_block(x_200v_2)
        x_200v_1 = pooling_block(x_200v_1)
        x_200v_2 = pooling_block(x_200v_2)
        x_200v = concatenate([x_200v_1, x_200v_2])
        block.append(x_200v)

    if which[1]:
        x_300v_1, x_300v_2 = embedding_block(
            sequence_input, 300, embedding_matrix_300v, trainable1=False, trainable2=True, max_features=max_features)
        x_300v_1 = conv1d_block(x_300v_1)
        x_300v_2 = conv1d_block(x_300v_2)
        x_300v_1 = pooling_block(x_300v_1)
        x_300v_2 = pooling_block(x_300v_2)
        x_300v = concatenate([x_300v_1, x_300v_2])
        block.append(x_300v)

    if which[2]:
        x_400v_1, x_400v_2 = embedding_block(
            sequence_input, 400, embedding_matrix_400v, trainable1=False, trainable2=True, max_features=max_features)
        x_400v_1 = conv1d_block(x_400v_1)
        x_400v_2 = conv1d_block(x_400v_2)
        x_400v_1 = pooling_block(x_400v_1)
        x_400v_2 = pooling_block(x_400v_2)
        x_400v = concatenate([x_400v_1, x_400v_2])
        block.append(x_400v)

    x = concatenate(block)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model
