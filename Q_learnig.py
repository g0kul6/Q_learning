import random
import numpy as np
import matplotlib.pyplot as plt
#class for q learning
class Q_learning:
    
    #getting the environment as input
    def __init__(self,env):
        self.env=env
    #intializing q table
    def intializing(self):
        Q_table=np.zeros((self.env.observation_space.n,self.env.action_space.n))
        return Q_table
    #training the agent
    def train(self,Q_table,alpha,gamma,epsi,decay,num_episode,max_steps):
        #getting intialized Q_table
        self.Q_table=Q_table
        #getting hyperparameter as input
        self.alpha=alpha 
        self.gamma=gamma
        self.epsi=epsi
        self.num_episode=num_episode
        self.max_steps=max_steps
        self.decay=decay
        max_epsilon=decay[0]
        min_epsilon=decay[1]
        decay_rate=decay[2]
        #storing all cumulative reward in train reward
        train_reward=[]
        penalty=[]
        for episode in range(num_episode):
            #reset the environment
            state=self.env.reset()
            cumilative_train_reward=0
            p=0
            for step in range(max_steps):
                #intialize random number to exploration-exploitation  trade off
                exp_vs_exp_tradeoff=random.uniform(0,1)
                #exploting the learned Q_table
                if exp_vs_exp_tradeoff>epsi:
                    action=np.argmax(Q_table[state,:])
                else:
                    #random action(exploration)
                    action=self.env.action_space.sample()
                new_state,reward,done,info=self.env.step(action)
                Q_table[state,action]=Q_table[state,action]+alpha*(reward+gamma*np.max(Q_table[new_state,:])-Q_table[state,action])
                cumilative_train_reward+=reward
                state=new_state
                if reward==-10:
                    p=p+1
                if done==True:
                    break
            #decreasing the epsilon so it explores less as the episodes increase
            penalty.append(p)
            epsi = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
            train_reward.append(cumilative_train_reward)
        return train_reward,penalty
    #ploting reward vs episodes
    def plot(self,reward,num_episodes,a):
        self.reward=reward
        self.a=a
        self.num_episodes=num_episodes
        x=range(num_episodes)
        plt.plot(x,reward)
        plt.xlabel('episodes')
        if a==0:
            plt.ylabel('cumulative_rewards')
        elif a==1:
             plt.ylabel('penalties')
        plt.show()
    #evaluation of agent
    def evaluate(self,Q_table,num_episodes,max_steps):
        self.Q_table=Q_table
        self.max_steps=max_steps
        self.num_episodes=num_episodes
        #storing all cumulative reward in eval_reward
        eval_rewards=[]
        for episode in range(num_episodes):
            state=self.env.reset()
            cumilative_eval_reward=0
            for step in range(max_steps):
                #taking the best action
                action=np.argmax(Q_table[state,:])
                new_state,reward,done,info=self.env.step(action)
                cumilative_eval_reward+=reward
                state=new_state
                if done==True:
                    break
            eval_rewards.append(cumilative_eval_reward)
        return eval_rewards 
    
    

    