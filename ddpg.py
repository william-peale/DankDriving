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


from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run

    env = gym.make("BipedalWalker-v2")
    print env.action_space
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 4  #Steering/Acceleration/Brake
    state_dim = 24  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 500000.
    episode_count = 10000
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


    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("Walker Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        
        ob = env.reset()

        s_t = np.array(ob).reshape((1,state_dim))
     
        total_reward = 0.
        for j in range(max_steps):
            loss = 0 
            epsilon = max(.1, epsilon - (1.0 / EXPLORE))
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])

            a_t_original = actor.model.predict(s_t.reshape((1,state_dim)))

            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.15, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.0 , 0.15, 0.30)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.0 , 0.15, 0.30)
            noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][3], 0.0 , 0.15, 0.30)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            a_t[0][3] = a_t_original[0][3] + noise_t[0][3]

            ob, r_t, done, info = env.step(a_t[0])
            if(i % 10 == 0 or train_indicator == 0):
                env.render()

            s_t1 = np.array(ob).reshape((1,state_dim))
            if j == max_steps - 1:
                r_t = -100
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0][0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3][0] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

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
    playGame(0)
