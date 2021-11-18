#importing necessary lib and Q_learning class
import gym
from Q_learnig import Q_learning
from config import number_info,max_steps,max_key,decay
import numpy as np
#getting the best hyperparameters
alpha=float(max_key.split(',')[0])
gamma=float(max_key.split(',')[1])
epsilon=float(max_key.split(',')[2])
#creating a environment of taxi-v3
env=gym.make("Taxi-v3")
#creating a abject of class Q_learning with taxi-v3 as its environment
obj=Q_learning(env)
#intializing the 
Q_table=obj.intializing()
#training the agent in the environment
train_rewards,penalty=obj.train(Q_table,alpha,gamma,epsilon,decay,number_info[0],max_steps)
#ploting cumulative train reward vs episodes
obj.plot(train_rewards,number_info[0],0)
obj.plot(penalty,number_info[0],1)
np.save('Q_table.npy',Q_table)
#evaluating
eval_rewards=obj.evaluate(Q_table,number_info[1],max_steps)
#ploting  cumulative eval reward vs episodes
obj.plot(eval_rewards,number_info[1],0)
#printing train and evaluation scores
print("train_score:",np.sum(train_rewards)/number_info[0])
print("eval_score:",np.sum(eval_rewards)/number_info[1])

