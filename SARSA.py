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

class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done):
        # TO DO: Add own code
        if not done:
            backup_target = r + self.gamma * self.Q_sa[s_next,a_next]
            self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (backup_target - self.Q_sa[s,a])
        pass

        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    i = 0
    #set the state to the initial environment state and take an initial step
    s = env.reset()
    a = pi.select_action(s, policy, epsilon, temp)
    while i <= n_timesteps:
        s_next, r, done = env.step(a)
        #model the next step to use in updating the Q-values on-policy
        a_next = pi.select_action(s_next, policy, epsilon, temp)
        #update the Q-values on-policy
        pi.update(s, a, r, s_next, a_next, done)
        if done:
            #reset the environment if the goal is reached and take an initial step again
            s = env.reset()
            a = pi.select_action(s, policy, epsilon, temp)
        else:
            #if goal is not reached, take another step
            s = s_next
            a = a_next
        if i%eval_interval == 0:
            #code for evaluation
            eval_timesteps.append(i)
            eval_returns.append(pi.Q_sa[s,a])
            if plot:
                env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution
        i += 1

    

    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 1000
    eval_interval = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    eval_returns, eval_timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns, eval_timesteps)
    
if __name__ == '__main__':
    test()
