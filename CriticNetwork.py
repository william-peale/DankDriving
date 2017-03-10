import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Convolution1D, MaxPooling2D, LSTM
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 512
HIDDEN2_UNITS = 512
HIDDEN3_UNITS = 512
LSTM_UNITS = 128
class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(80,450,10)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(80,450,10)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self,img_size,lidar_inputs,num_images):
        
        print("Now we build the model")

        #ACTION MODEL

        action_dim = 1
        
        #S0 = Input(shape=(num_images,img_size,img_size))
        #c0 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(S0)
        #c1 = Convolution2D(32, 5, 5, border_mode='same', activation='relu')(c0)
        #p0 = MaxPooling2D(pool_size=(2,2))(c1)
        #f0 = Flatten()(p0)
        #h2 = Dense(HIDDEN2_UNITS, activation='linear')(f0)
        
        #Lidar Input
        S1 = Input(shape=(num_images,lidar_inputs))
        c0 = Convolution1D(100, 5, border_mode='same',activation='relu')(S1)
        l0 = LSTM(LSTM_UNITS,activation='relu')(c0)
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(l0)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(h0)
        
        #m0 = merge([h1,h2], mode='sum')
        #h3 = Dense(HIDDEN3_UNITS, activation='relu')(m0)

        #STATE MODEL
          
        A = Input(shape=[2],name='action2')   
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A) 
        h4 = merge([h1,a1],mode='sum')    
        h5 = Dense(HIDDEN2_UNITS, activation='relu')(h4)
        V = Dense(1,activation='linear')(h5)   
        model = Model(input=[S1,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S1 
