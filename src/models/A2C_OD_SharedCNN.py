import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import Model, layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, LSTM, TimeDistributed, ConvLSTM2D, Reshape, Conv2DTranspose, UpSampling2D, ReLU

class ProbabilityDistribution(Model):
    def call(self, logits, **kwargs):
        # sample a categorical action given logits
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        neglogprob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=action)
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
        return inputs + out

class Model(Model):
    def __init__(self, ac_dim):
        super(Model, self).__init__()
        self.model_name = 'A2C_SharedCNN_OD'
        self.conv1 = Conv2D(16, 3, padding='same', kernel_regularizer=regularizers.l2(0.000))
        self.conv2 = Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.000))
        self.conv3 = Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.000))
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
        self.d1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.000))
        self.d4 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.dist = ProbabilityDistribution()
        self.logits = Dense(ac_dim, name='policy_logits', kernel_regularizer=regularizers.l2(0e-3))
        self.value = Dense(1, name='value', kernel_regularizer=regularizers.l2(0e-3))

        self.d7 = Dense(16*16*16, activation='relu', kernel_regularizer=regularizers.l2(0.000))
        self.reshape = Reshape((16,16,16)) 
        self.deconv1 = Conv2DTranspose(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0e-3))
        self.deconv2 = Conv2DTranspose(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(0e-3))
        self.deconv3 = Conv2DTranspose(3, 3, padding='same', kernel_regularizer=regularizers.l2(0e-3))
        self.upsample1 = UpSampling2D(2)
        self.upsample2 = UpSampling2D(2)
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
        flattened = self.relu(flattened)
        flattened = self.d1(flattened)

        # actor dense layers
        #x = self.d1(flattened)

        # critic dense layers
        #y = self.d4(flattened)

        # observation deconvolution
        #z = self.d7(tf.concat([flattened,self.logits(flattened)],axis=-1))
        z = self.d7(flattened)
        z = self.reshape(z) 
        z = self.deconv1(z)
        z = self.upsample1(z)
        z = self.deconv2(z) 
        z = self.upsample2(z)
        z = self.deconv3(z) 

        return self.logits(flattened), self.value(flattened), tf.expand_dims(z, axis=-1)

    def action_value(self, obs):
        logits, value, _ = self.predict_on_batch(obs)
        action, _ = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    def action_value_neglogprobs_obs(self, obs):
        logits, value, obs = self.predict_on_batch(obs)
        action, neglogprob = self.dist.predict_on_batch(logits)
        #action = tf.random.categorical(logits,1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1), neglogprob, np.squeeze(obs, axis=-1)
