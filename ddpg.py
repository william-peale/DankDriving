import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf

import json
import gym
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
import time
import vrep

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run

    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.995
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 1  #Steering/Acceleration/Brake
    state_dim = 32  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 100000
    max_steps = 2000
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

    print ('Startin the vrep')
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
    if clientID!=-1:
        print ('Im connected m8')

    #get object handles
    robotCollision=vrep.simxGetCollisionHandle(clientID,"swag3",vrep.simx_opmode_blocking)
    print robotCollision
    motorFrontLeft=vrep.simxGetObjectHandle(clientID,"driving_joint_front_right",vrep.simx_opmode_blocking)
    motorFrontRight=vrep.simxGetObjectHandle(clientID,"driving_joint_front_left",vrep.simx_opmode_blocking)
    motorRearLeft=vrep.simxGetObjectHandle(clientID,"driving_joint_rear_right",vrep.simx_opmode_blocking)
    motorRearRight=vrep.simxGetObjectHandle(clientID,"driving_joint_rear_left",vrep.simx_opmode_blocking)
    steeringWheelLeft=vrep.simxGetObjectHandle(clientID,"steering_joint_fl",vrep.simx_opmode_blocking)
    steeringWheelRight=vrep.simxGetObjectHandle(clientID,"steering_joint_fr",vrep.simx_opmode_blocking)
    sensor_handle=vrep.simxGetObjectHandle(clientID,"Vision_sensor",vrep.simx_opmode_blocking)[1]
    proximitySensor1=vrep.simxGetObjectHandle(clientID,"Proximity_sensor",vrep.simx_opmode_blocking)[1]
    proximitySensor2=vrep.simxGetObjectHandle(clientID,"Proximity_sensor0",vrep.simx_opmode_blocking)[1]
    proximitySensor3=vrep.simxGetObjectHandle(clientID,"Proximity_sensor1",vrep.simx_opmode_blocking)[1]
    proximitySensor4=vrep.simxGetObjectHandle(clientID,"Proximity_sensor2",vrep.simx_opmode_blocking)[1]
    proximitySensor5=vrep.simxGetObjectHandle(clientID,"Proximity_sensor3",vrep.simx_opmode_blocking)[1]
    proximitySensor6=vrep.simxGetObjectHandle(clientID,"Proximity_sensor4",vrep.simx_opmode_blocking)[1]
    proximitySensor7=vrep.simxGetObjectHandle(clientID,"Proximity_sensor5",vrep.simx_opmode_blocking)[1]
    proximitySensorFinal=vrep.simxGetObjectHandle(clientID,"Proximity_sensor6",vrep.simx_opmode_blocking)[1]
    # enable the synchronous mode on the client:
    vrep.simxSynchronous(clientID,True)
    
    #Now load the weight
    print("Now we load the weight")
    try:
        #actor.model.load_weights("actormodel.h5")
        #critic.model.load_weights("criticmodel.h5")
        #actor.target_model.load_weights("actormodel.h5")
        #critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("imma start drivin.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))


        visited = [False,False,False,False,False,False,False]
        total_reward = 0
        for j in range(max_steps):
            
            proxArr = [False,False,False,False,False,False,False]
            #Start up vrep
            vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
            
            #Poll initially to do stuff
            _, _, s_t = vrep.simxGetVisionSensorDepthBuffer(clientID, sensor_handle, vrep.simx_opmode_streaming)
            _, _, _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor1, vrep.simx_opmode_streaming)
            _, _, _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor2, vrep.simx_opmode_streaming)
            _, _, _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor3, vrep.simx_opmode_streaming)
            _, _, _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor4, vrep.simx_opmode_streaming)
            _, _, _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor5, vrep.simx_opmode_streaming)
            _, _, _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor6, vrep.simx_opmode_streaming)
            _, _, _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor7, vrep.simx_opmode_streaming)
            _, _, _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensorFinal, vrep.simx_opmode_streaming)
            _, _=vrep.simxReadCollision(clientID,robotCollision[1],vrep.simx_opmode_streaming)
            
            vrep.simxSynchronousTrigger(clientID);
            
            _, _, s_t = vrep.simxGetVisionSensorDepthBuffer(clientID, sensor_handle, vrep.simx_opmode_buffer)
            
            s_t = np.array(s_t).reshape((1, state_dim))
            loss = 0 
            epsilon = max(.1, epsilon - (1.0 / EXPLORE))
            a_t = 0
            noise_t = 0

            a_t_original = actor.model.predict(s_t.reshape((1,state_dim)))

            noise_t = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.15, 0.30)
            #noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.0 , 0.15, 0.30)
            #noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.0 , 0.15, 0.30)
            #noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][3], 0.0 , 0.15, 0.30)

            a_t = a_t_original + noise_t
            #a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            #a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            #a_t[0][3] = a_t_original[0][3] + noise_t[0][3]

            #ob, r_t, done, info = env.step(a_t[0]) kys openai old stuff

            #READ FROM VREP HERE MIGHT BE MESSY

            vrep.simxSynchronousTrigger(clientID); #IM TRIGGERED


            r_t = -.1
            done = False

            vrep.simxSetJointTargetVelocity(clientID,motorFrontLeft[1],-2,vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(clientID,motorFrontRight[1],-2,vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(clientID,motorRearLeft[1],-2,vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(clientID,motorRearRight[1],-2,vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetPosition(clientID,steeringWheelLeft[1],a_t[0],vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetPosition(clientID,steeringWheelRight[1],a_t[0],vrep.simx_opmode_blocking)
            
            _, _, ob = vrep.simxGetVisionSensorDepthBuffer(clientID, sensor_handle, vrep.simx_opmode_buffer)
            _, proxArr[0], _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor1, vrep.simx_opmode_buffer)
            _, proxArr[1], _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor2, vrep.simx_opmode_buffer)
            _, proxArr[2], _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor3, vrep.simx_opmode_buffer)
            _, proxArr[3], _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor4, vrep.simx_opmode_buffer)
            _, proxArr[4], _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor5, vrep.simx_opmode_buffer)
            _, proxArr[5], _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor6, vrep.simx_opmode_buffer)
            _, proxArr[6], _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensor7, vrep.simx_opmode_buffer)
            _, finished, _, _, _ = vrep.simxReadProximitySensor(clientID, proximitySensorFinal, vrep.simx_opmode_buffer)
            _, collided = vrep.simxReadCollision(clientID,robotCollision[1],vrep.simx_opmode_buffer)
            for i in range(len(proxArr)):
                if proxArr[i] and not visited[i]:
                    r_t += 5
                    visited[i] = True
                    
            if finished:
                done = True
                r_t += 100
            if collided:
                done = True
                r_t += -50
           

            s_t1 = np.array(ob).reshape((1,state_dim))
            buff.add(s_t, a_t, r_t, s_t1, done)      #Add replay buffer
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0][0] for e in batch])
            actions = np.asarray([e[1][0] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3][0] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1][0] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            #print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        reward_history.append(total_reward)
        if(len(reward_history) > 50):
            r_sum = 0
            for i_prime in range(len(reward_history)-50,len(reward_history)):
                r_sum += reward_history[i_prime]
            reward_avg_history.append(float(r_sum)/50.0)
        y = [x for x in range(len(reward_history))]
        y1 = [x for x in range(len(reward_history)) if(x >= 50)]
	plt.plot(y,reward_history)
        plt.plot(y1,reward_avg_history)
	plt.savefig("graph.png")
	plt.clf()
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward) + " Epsilon: " + str(epsilon))
        print("Total Step: " + str(step))
        print("")

    print("Finish.")

if __name__ == "__main__":
    playGame(1)
