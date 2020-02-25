import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import Model, layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, LSTM, TimeDistributed, ConvLSTM2D, ReLU

class ProbabilityDistribution(Model):
    def call(self, logits, **kwargs):
        # sample a categorical action given logits
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        neglogprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action)
        return action, neglogprob

class ResidualBlock(layers.Layer):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        
    def build(self, input_shape):
        self.relu = layers.ReLU()
        self.conv1 = Conv2D(input_shape[-1], 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.000))
        self.conv2 = Conv2D(input_shape[-1], 3, padding='same', kernel_regularizer=regularizers.l2(0.000))

    def call (self, inputs):
        x = self.relu(inputs)
        x = self.conv1(x)
        out = self.conv2(x)
        return tf.keras.layers.add([inputs, out])

class Model(Model):
    def __init__(self, ac_dim):
        super(Model, self).__init__()
        self.model_name = 'A2C_SharedCNN'
        self.conv1 = Conv2D(16, 3, padding='same', kernel_regularizer=regularizers.l2(0.000))#, kernel_initializer='he_uniform')
        self.conv2 = Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.000))#, kernel_initializer='he_uniform')
        self.conv3 = Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.000))#, kernel_initializer='he_uniform')
        self.maxpool1 = MaxPool2D(pool_size=3, strides=2, padding='same')
        self.maxpool2 = MaxPool2D(pool_size=3, strides=2, padding='same')
        self.maxpool3 = MaxPool2D(pool_size=3, strides=2, padding='same')
        self.res1 = ResidualBlock()
        self.res2 = ResidualBlock()
        self.res3 = ResidualBlock()
        self.res4 = ResidualBlock()
        self.res5 = ResidualBlock()
        self.res6 = ResidualBlock()
        self.flatten = Flatten()
        self.d1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0e-3))
        self.d4 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0e-3))
        self.dist = ProbabilityDistribution()
        self.logits = Dense(ac_dim, name='policy_logits', kernel_regularizer=regularizers.l2(0e-3))
        self.value = Dense(1, name='value', kernel_regularizer=regularizers.l2(0e-3))
        self.relu = ReLU()

    def call(self, inputs):
        i = self.conv1(inputs)
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
        #flattened = self.relu(flattened)

        # actor dense layers
        x = self.d1(flattened)
        logits = self.logits(x)

        # critic dense layers
        y = self.d4(flattened)
        value = self.value(y)

        return logits, value  

    def action_value(self, obs):
        logits, value = self.predict_on_batch(obs)
        action, _ = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    def action_value_neglogprob(self, obs):
        logits, value = self.predict_on_batch(obs)
        action, neglogprob = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1), neglogprob
