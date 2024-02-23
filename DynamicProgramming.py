#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        #Greedy policy: choose the action that gives the highest Q-value for the current step
        a = argmax(self.Q_sa[s,:])
        #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        #Bellman optimality equation, code implementation based on the one on slide 121 in lecture 2
        self.Q_sa[s,a] = np.sum(p_sas * (r_sas + self.gamma * np.max(self.Q_sa, axis=1)))
        pass
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)
    
    
    S = env.n_states
    A = env.n_actions
    max_err = threshold + 1
    i = 0
    #Iterate until convergence
    while max_err > threshold:
        max_err = 0
        #Iterate over all actions possible in each state
        for s in range(S):
            for a in range(A):
                #Get transition function and reward function from the environment class
                p_sas, r_sas = env.model(s,a)
                x = QIagent.Q_sa[s,a]
                #Update the Q matrix with the current action+state
                QIagent.update(s,a,p_sas,r_sas)
                err = np.abs(x - QIagent.Q_sa[s,a])
                #Update the Maximal error
                max_err = max(max_err, err)
        # Plot current Q-value estimates & print max error
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
        print("Q-value iteration, iteration {}, max error {}".format(i,max_err))
        i += 1

    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)
    
    # view optimal policy
    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=1)
        s = s_next

    #V_star is the converged optimal value at the start -> it's the expected reward value from taking the path with that value
    V_star = max(QIagent.Q_sa[3,:])
    print("Converged optimal value at the start: {}".format(V_star))
    mean_number_of_steps = (V_star - env.goal_rewards[0])/env.reward_per_step
    print("Mean number of steps under optimal policy: {}".format(mean_number_of_steps))
    mean_reward_per_timestep = V_star/mean_number_of_steps
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()
