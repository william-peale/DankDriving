import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
#from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 128
HIDDEN3_UNITS = 128

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(80,32,3)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(80,32,3) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, img_size,lidar_inputs,num_images):
        print("Now we build the model")

        #Input image
        #S0 = Input(shape=(num_images,img_size,img_size))
        #c0 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(S0)
        #c1 = Convolution2D(32, 5, 5, border_mode='same', activation='relu')(c0)
        #p0 = MaxPooling2D(pool_size=(2,2))(c1)
        #f0 = Flatten()(p0)
        #h2 = Dense(HIDDEN2_UNITS, activation='linear')(f0)
        
        #Lidar Input
        S1 = Input(shape=[lidar_inputs])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S1)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(h0)
        
        #m0 = merge([h1,h2], mode='sum')
        #h3 = Dense(HIDDEN3_UNITS, activation='relu')(m0)
                   
        #Output1 = Dense(1,activation='sigmoid')(h1)
        #Output2 = Dense(1,activation='tanh')(h1)
        #V = merge([Output1,Output2],mode='concat')
        V = Dense(1,activation='tanh')(h1)
        
        model = Model(input=[S1],output=V)
        return model, model.trainable_weights, S1

