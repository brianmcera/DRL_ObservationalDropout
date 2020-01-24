import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization

class ProbabilityDistribution(Model):
    def call(self, logits, **kwargs):
        # sample a categorical action given logits
        print(logits)
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(Model):
    def __init__(self, ac_dim):
        super(Model, self).__init__()
        self.conv1 = Conv2D(16, 2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv2 = Conv2D(16, 2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv3 = Conv2D(32, 2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.maxpool1 = MaxPool2D(pool_size=(2,2))
        self.maxpool2 = MaxPool2D(pool_size=(2,2))
        self.maxpool3 = MaxPool2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.d1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d2 = Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d3 = Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d4 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d5 = Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d6 = Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.logits = Dense(ac_dim, name='policy_logits')
        self.value = Dense(1, name='value')
        self.dropout1 = Dropout(rate=0.5)
        self.dropout2 = Dropout(rate=0.5)
        self.dropout3 = Dropout(rate=0.5)
        self.dropout4 = Dropout(rate=0.5)
        self.batchnormalization1 = BatchNormalization()
        self.batchnormalization2 = BatchNormalization()
        self.batchnormalization3 = BatchNormalization()
        self.batchnormalization4 = BatchNormalization()
        self.batchnormalization5 = BatchNormalization()
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        i = tf.convert_to_tensor(inputs)
        #i = self.maxpool1(i)
        i = self.batchnormalization1(i)
        i = self.conv1(i)
        #i = self.maxpool2(i)
        i = self.batchnormalization2(i)
        i = self.conv2(i)
        #i = self.maxpool3(i)
        i = self.batchnormalization3(i)
        i = self.conv3(i)
        flattened = self.flatten(i)

        # actor dense layers
        x = self.d1(flattened)
        x = self.d2(x)
        x = self.dropout1(x)
        x = self.batchnormalization4(x)
        x = self.d3(x)
        x = self.dropout2(x)

        # critic dense layers
        y = self.d4(flattened)
        y = self.d5(y)
        y = self.dropout3(y)
        y = self.batchnormalization5(y)
        y = self.d6(y)
        y = self.dropout4(y)
        return self.logits(x), self.value(y)

    def action_value(self, obs):
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
