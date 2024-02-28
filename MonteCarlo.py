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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        T = len(actions)
        backup_target = 0
        for t in reversed(range(T)):
            backup_target = rewards[t] + self.gamma * backup_target
            self.Q_sa[states[t],actions[t]] = self.Q_sa[states[t],actions[t]] + self.learning_rate * (backup_target - self.Q_sa[states[t],actions[t]])

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    i = 0
    while i < n_timesteps:
        #Setup variables for this episode
        s = env.reset()
        a_ep = []
        s_ep = [s]
        r_ep = []
        for t in range(max_episode_length):
            #Gather the values for this episode
            a = pi.select_action(s_ep[t],policy, epsilon, temp)
            s_next, r, done = env.step(a)
            a_ep.append(a)
            s_ep.append(s_next)
            r_ep.append(r)
            i += 1
            if done:
                break
            if i%eval_interval == 0 :
                #Code for evaluation
                eval_timesteps.append(i)
                eval_returns.append(pi.evaluate(eval_env))
                if plot:
                    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution
        pi.update(s_ep, a_ep, r_ep)
    #Ensures all runs have the same number of evaluations, important in the case of averaging over multiple runs
    if len(eval_returns) != (int(n_timesteps/eval_interval)):
        eval_returns = np.pad(eval_returns, int(n_timesteps/eval_interval)-len(eval_returns),constant_values=eval_returns[-1])
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, eval_interval=100)
    print(eval_returns, eval_timesteps)
    
    
            
if __name__ == '__main__':
    test()
