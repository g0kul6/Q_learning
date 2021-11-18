#importing necessary classes and librarary
from Q_learnig import Q_learning
import numpy as np
import gym
env=gym.make("Taxi-v3")
#intializing hyperparameters and model parameters:
#discount constant
alpha=[0.7,0.5,0.6,0.9]
#discount rate
gamma=[0.6,0.7,0.5,0.8]
#epsilon exploration vs eploitation hyperpararmeter
epsilon=[0.4,0.2,0.3,1.0]
max_epsilon = 1.0             
min_epsilon = 0.01            
decay_rate = 0.01 
decay=[max_epsilon,min_epsilon,decay_rate]     
#number episodes in  train,evaluation,gridsearch and max no of steps in episode
train_episode=20000
grid_episode=2000
eval_episode=100
max_steps=100
number_info=[train_episode,eval_episode,grid_episode]
#position of max alpha,gamma 
pos_alpha_max=0
pos_gamma_max=0
#hyperparameter tuning
tune=Q_learning(env)
Q_table=tune.intializing()
reward_dict={}
#grid search
for i in range(len(alpha)):
    for j in range(len(gamma)):
        for k in range(len(epsilon)):
            train_reward=tune.train(Q_table,alpha[i],gamma[j],epsilon[k],decay,number_info[2],max_steps)
            a=str(alpha[i])+','+str(gamma[j])+','+str(epsilon[k])
            reward_dict[a]=np.sum(train_reward)/number_info[0]
#getting the key of the maximum score given by a set of hyperparameters
max_key=max(reward_dict, key=reward_dict. get) 
