## this script includes possible deep models for e2edl

from keras.models import Sequential
from keras.models import Model
from keras import regularizers
from keras.layers.recurrent import LSTM
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten

def baseline_dense_model(X, Y, vtag=2):

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

    return model


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

    

def convolutional_model(X, Y, vtag=2):


    batch_size = 32 # in each iteration, we consider 32 training examples at once
    num_epochs = 200 # we iterate 200 times over the entire training set
    kernel_size = 3 # we will use 3x3 kernels throughout
    pool_size = 2 # we will use 2x2 pooling throughout
    conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
    conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
    drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
    drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
    hidden_size = 512 # the FC layer will have 512 neurons

    inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)
    # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    drop_2 = Dropout(drop_prob_1)(pool_2)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)
    out = Dense(num_classes, activation='softmax')(drop_3)
    
    model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
                  optimizer='adam', # using the Adam optimiser
                  metrics=['accuracy']) # reporting the accuracy

    model.fit(X_train, Y_train,                # Train the model using the training set...
              batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

    ## do 1D convolutions
    ## this is less intense
    
    pass

def convolutional_ae_model(X, Y, vtag=2):

    ## first auto-encode the data, then do convolutions
    
    pass
