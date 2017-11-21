## this script includes possible deep models for e2edl

from keras.models import Sequential
from keras.models import Model
from keras import regularizers
from keras.layers.recurrent import LSTM
from keras.layers import Input, Conv1D, MaxPool1D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
import numpy as np

def baseline_dense_model(X, Y, xtest ,vtag=2):

    inshape = int(X.shape[1])
    outshape = int(Y.shape[1])
    layer_first = int(inshape*0.75)
    layer_second = int(inshape*0.5)
    layer_third = int(inshape*0.3)
    layer_fourth = int(outshape*2)

    model = Sequential()
    model.add(Dense(inshape, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(layer_first, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(layer_first, activation='relu'))
    model.add(Dense(layer_first, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(layer_second,activation='relu'))
    model.add(Dense(layer_third,activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dropout(0.15))
    model.add(Dense(layer_fourth,activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(layer_fourth,activation='relu'))
    model.add(Dense(layer_fourth,activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(layer_fourth,activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(outshape, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Fit the model
    model.fit(X, Y, epochs=300, batch_size=60, verbose=vtag)

    return model.predict(xtest)


def autoencoder_model(X, Y,vtag=2):

    inshape = int(X.shape[1])
    outshape = int(Y.shape[1])

    ## autoencoder
    encoding_dim = int(inshape/2)
    
    # this is our input placeholder
    input_matrix = Input(shape=(inshape,))
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5),name="encoded_layer")(input_matrix)
    decoded = Dense(inshape, activation='sigmoid')(encoded)


    # this model maps an input to its reconstruction
    autoencoder = Model(input_matrix, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X, X,
                    epochs=300,
                    batch_size=80,
                    shuffle=True,
                    verbose=vtag)


    ## train on a representation - more efficient
    l2 = int(encoding_dim/2)
    l3 = int(encoding_dim/3)
    l4 = int(encoding_dim/4)
    inputs_2 = Input(shape=(inshape,))
    embedded_layer =  Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5),name="encoded_layer")(inputs_2)
    dropout1 = Dropout(0.2)(embedded_layer)
    dense2 = Dense(l2,activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    dense3 = Dense(l3,activation='relu')(dropout2)
    dropout3 = Dropout(0.2)(dense3)
    dense3 = Dense(l4,activation='relu')(dropout3)
    outlayer = Dense(outshape,activation="sigmoid")(dense3)
    predictor = Model(inputs_2,outlayer)

    predictor.compile(optimizer='adam', loss='binary_crossentropy')
    predictor.get_layer('encoded_layer').set_weights(autoencoder.get_layer('encoded_layer').get_weights())
    
    predictor.fit(X, Y,
                  epochs=100,
                  batch_size=60,
                  shuffle=True,
                  verbose=vtag)
    
    return predictor

    

def convolutional_model(X, Y, x_test, vtag=2):

    nfeat= int(X.shape[1])
    tshape = int(Y.shape[1])
    scaler = StandardScaler()
    
    inp =  Input(shape=(nfeat, 1))
    conv = Conv1D(filters=15, kernel_size=5)(inp)
    pool = MaxPool1D(pool_size=4)(conv)
    conv2 = Conv1D(filters=15, kernel_size=5)(pool)
    flat = Flatten()(conv2)
    dense1 = Dense(tshape)(flat)
    d1 = Dropout(0.1)(dense1)
    dense2 = Dense(tshape)(d1)
    model = Model(inp, dense2)
    model.compile(loss='mse', optimizer='adam')

    # get some data
    X = np.expand_dims(X, axis=2)

    # fit model
    model.fit(X, Y,
              epochs=350,
              batch_size=60,
              shuffle=True,
              verbose=vtag)

    xtest = np.expand_dims(x_test,axis=2)
    return model.predict(xtest)

def convolutional_ae_model(X, Y, vtag=2):

    ## first auto-encode the data, then do convolutions
    
    pass
