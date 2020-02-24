import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import Model, layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, LSTM, TimeDistributed, ConvLSTM2D, ReLU

class ProbabilityDistribution(Model):
    def call(self, logits, **kwargs):
        # sample a categorical action given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class ResidualBlock(layers.Layer):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        
    def build(self, input_shape):
        self.relu = layers.ReLU()
        self.conv1 = Conv2D(input_shape[-1], 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))
        self.conv2 = Conv2D(input_shape[-1], 3, padding='same', kernel_regularizer=regularizers.l2(0.001))

    def call (self, inputs):
        x = self.relu(inputs)
        x = self.conv1(x)
        out = self.conv2(x)
        return inputs + out

class Model(Model):
    def __init__(self, ac_dim):
        super(Model, self).__init__()
        self.conv1 = Conv2D(16, 3, padding='same', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_uniform')
        self.conv2 = Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_uniform')
        self.conv3 = Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_uniform')
        self.maxpool1 = MaxPool2D(pool_size=3, strides=2)
        self.maxpool2 = MaxPool2D(pool_size=3, strides=2)
        self.maxpool3 = MaxPool2D(pool_size=3, strides=2)
        self.res1 = ResidualBlock()
        self.res2 = ResidualBlock()
        self.res3 = ResidualBlock()
        self.res4 = ResidualBlock()
        self.res5 = ResidualBlock()
        self.res6 = ResidualBlock()
        self.flatten = Flatten()


        self.conv1C = Conv2D(16, 3, padding='same', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_uniform')
        self.conv2C = Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_uniform')
        self.conv3C = Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_uniform')
        self.maxpool1C = MaxPool2D(pool_size=3, strides=2)
        self.maxpool2C = MaxPool2D(pool_size=3, strides=2)
        self.maxpool3C = MaxPool2D(pool_size=3, strides=2)
        self.res1C = ResidualBlock()
        self.res2C = ResidualBlock()
        self.res3C = ResidualBlock()
        self.res4C = ResidualBlock()
        self.res5C = ResidualBlock()
        self.res6C = ResidualBlock()
        self.flattenC = Flatten()


        self.d1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.d2 = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.d3 = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.d4 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.d5 = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.d6 = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.dropout1 = Dropout(rate=0.5)
        self.dropout2 = Dropout(rate=0.5)
        self.dropout3 = Dropout(rate=0.5)
        self.dropout4 = Dropout(rate=0.5)
        self.batchnormalization0 = BatchNormalization()
        self.batchnormalization1 = BatchNormalization()
        self.batchnormalization2 = BatchNormalization()
        self.batchnormalization3 = BatchNormalization()
        self.batchnormalization4 = BatchNormalization()
        self.batchnormalization5 = BatchNormalization()
        self.LSTM1 = LSTM(20, return_sequences=False, stateful=False)
        self.LSTM2 = LSTM(20, return_sequences=False, stateful=False)
        self.dist = ProbabilityDistribution()
        self.logits = Dense(ac_dim, name='policy_logits')
        self.value = Dense(1, name='value')
        self.relu = ReLU()
        self.reluC = ReLU()

    def call(self, inputs):
        # actor model
        i = tf.convert_to_tensor(inputs)
        i = self.conv1(i)
        i = self.maxpool1(i)
        i = self.res1(i)
        i = self.res2(i)
        i = self.conv2(i)
        i = self.maxpool2(i)
        i = self.res3(i)
        i = self.res4(i)
        i = self.conv3(i)
        i = self.maxpool3(i)
        i = self.res5(i)
        i = self.res6(i)
        flattened = self.flatten(i)
        flattened = self.relu(flattened)
        x = self.d1(flattened)

        # critic dense layers
        j = tf.convert_to_tensor(inputs)
        j = self.conv1C(j)
        j = self.maxpool1C(j)
        j = self.res1C(j)
        j = self.res2C(j)
        j = self.conv2C(j)
        j = self.maxpool2C(j)
        j = self.res3C(j)
        j = self.res4C(j)
        j = self.conv3C(j)
        j = self.maxpool3C(j)
        j = self.res5C(j)
        j = self.res6C(j)
        flattened2 = self.flattenC(j)
        flattened2 = self.reluC(flattened2)
        y = self.d4(flattened2)

        return self.logits(x), self.value(y)

    def action_value(self, obs):
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        #action = tf.random.categorical(logits,1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)