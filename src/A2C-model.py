import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization

class ProbabilityDistribution(Model):
    def call(self,inputs):
        # sample a categorical action given logits
        return tf.squeeze(tf.random.categorical(logits,1), axis=-1)

class A2CModel(Model):
    def __init__(self, ob_dim, ac_dim):
        super(actor, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv2 = Conv2D(32, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv3 = Conv2D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.maxpool1 = MaxPool2D(pool_size=(2,2), input_shape=ob_dim)
        self.maxpool2 = MaxPool2D(pool_size=(2,2))
        self.maxpool3 = MaxPool2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.d1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.d2 = Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.d3 = Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.d4 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.d5 = Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.d6 = Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.logits = Dense(ac_dim, name='policy_logits')
        self.value = Dense(1, name='value')
        self.dropout = Dropout(rate=0.5)
        self.batchnormalization1 = BatchNormalization()
        self.batchnormalization2 = BatchNormalization()
        self.batchnormalization3 = BatchNormalization()
        self.dist = ProbablityDistribution()

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        x = self.maxpool1(x)
        x = self.conv1(x)
        x = self.maxpool2(x)
        x = self.conv2(x)
        x = self.maxpool3(x)
        x = self.conv3(x)
        x = self.batchnormalization1(x)
        flattened = self.flatten(x)
        # actor dense layers
        x = self.d1(flattened)
        x = self.batchnormalization2(x)
        x = self.d2(x)
        x = self.dropout(x)
        x = self.d3(x)
        # critic dense layers
        y = self.d4(flattened)
        y = self.batchnormalization3(y)
        y = self.d5(y)
        y = self.dropout(y)
        y = self.d6(y)
        return self.logits(x), self.value(y)
