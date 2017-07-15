import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf

import json


import sys
import time


from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run

    BUFFER_SIZE = 1000000
    BATCH_SIZE = 32
    GAMMA = 0.995
    TAU = 0.001    #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.002     #Lerning rate for Critic

    temporal_dim = 3
    action_dim = 2  #Steering/Acceleration/Brake
    state_dim = 450  #of sensors input
    
    vision = False

    EXPLORE = 100000.
    episode_count = 100000
    max_steps = 999999999999
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    reward_history = []
    reward_avg_history = []
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    

    #Now load the weight
    print("Now we load the weight")
    try:
        #UNCOMMENT THIS TO LOAD WEIGHTS
        actor.model.load_weights("actormodel_best.h5")
        critic.model.load_weights("criticmodel_best.h5")
        actor.target_model.load_weights("actormodel_best.h5")
        critic.target_model.load_weights("criticmodel_best.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("imma start drivin.")
    max_avg_reward = -99999
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))


        total_reward = 0
        for j in range(max_steps):
            
            #Start up vrep
            
            s_t = #PUT ALL THAT LIDAR SHIT IN HERE
            loss = 0 
            epsilon = max(.15, epsilon - (1.0 / EXPLORE))
            a_t = np.zeros([1,2])
            noise_t = np.zeros([1,2])

            a_t_original = actor.model.predict(s_t)

            noise_t[0][0] = train_indicator * max(epsilon, 0.15) * OU.function(a_t_original[0][0],  0.7 , 1.0, 0.3)
            noise_t[0][1] = train_indicator * max(epsilon, 0.15) * OU.function(a_t_original[0][1],  0.0 , 0.60, 0.60)
            
            #noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.0 , 0.15, 0.30)
            #noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][3], 0.0 , 0.15, 0.30)

            a_t[0][0] = a_t_original[0][0] #+ noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] #+ noise_t[0][1]
            #a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            #a_t[0][3] = a_t_original[0][3] + noise_t[0][3]
            if(a_t[0][1] > 1):
                a_t[0][1] = 1
            if(a_t[0][1] < -1):
                a_t[0][1] = -1
            #ob, r_t, done, info = env.step(a_t[0]) kys openai old stuff

            #READ FROM VREP HERE MIGHT BE MESSY

            r_t = -1.0
            done = False
            print "Throttle " + str(a[0][0])
            print "Steering " + str(a[0][1])
            
            s_t1 = np.array(s_t_arr).reshape((1,temporal_dim,state_dim))
            #Do the batch update
           
            total_reward += r_t
            s_t = s_t1
        
            #print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
         print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward) + " Epsilon: " + str(epsilon))
        print("Total Step: " + str(step))
        print("")

    print("Finish.")

if __name__ == "__main__":
    playGame(0)
