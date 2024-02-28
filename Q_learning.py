#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        # TO DO: Add own code
        if not done:
            backup_target = r + self.gamma * max(self.Q_sa[s_next,:])
            self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (backup_target - self.Q_sa[s,a])
        pass

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    
    #set the state to the starting position defined in the environment
    s = env.reset()
    for i in range(n_timesteps):
        #select the next action according to the chosen policy
        a = agent.select_action(s, policy, epsilon, temp)
        s_next, r, done = env.step(a)
        #update the Q-value for this step
        agent.update(s,a,r,s_next,done)
        if done:
            #reset the state to the starting position if goal is reached
            s = env.reset()
        else:
            #if goal is not reached, take another step
            s = s_next
        if i%eval_interval == 0:
            #code for evaluation
            eval_timesteps.append(i)
            eval_returns.append(agent.evaluate(eval_env))
            if plot:
                env.render(Q_sa=agent.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution


    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 1000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'softmax' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
