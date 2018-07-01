from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import backend as K

class Transpose(Layer):
    def call(self, x):
        x = K.permute_dimensions(x, pattern=[0,2,1])
        return x
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[1])
    
def sentiment_analysis(embedding_matrix,maxlen=64, embed_size=400, max_features=4000):
    sequence_input = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable = True)(sequence_input)

    x = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size = 1, padding = "valid", activation='relu', kernel_initializer = "glorot_uniform")(x)
    x = Conv1D(64, kernel_size=1, activation='relu', padding = "valid", kernel_initializer = "glorot_uniform")(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding = "valid", kernel_initializer = "glorot_uniform")(x)

    x1 = Transpose()(x)
    x1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x1)
    x1 = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x1)
    x1 = Flatten()(x1)
    
    x2 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x2 = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x2)
    x2 = Flatten()(x2)
    
    x = Concatenate(axis=-1)([x1,x2])
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model