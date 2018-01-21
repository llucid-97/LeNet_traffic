import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Lambda, BatchNormalization, Softmax
import keras
import keras.backend as K

inputs = Input(shape=(32, 32, 3))

prep = Lambda(lambda x: tf.image.rgb_to_grayscale(x))(inputs)
prep_whitening = Lambda(lambda x: (x - 128) / 255)(prep)
prep_batchNorm = BatchNormalization()(prep)

net = keras.layers.Flatten()(prep_whitening)
net = Dense(43)(net)
net = Softmax()(net)
net = 

LeNet = Model(inputs=inputs, outputs=net)
layerProbe = Model(inputs=inputs, outputs=prep_whitening)


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


LeNet.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy', mean_pred])