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

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n, t):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        backup_target = 0
        if not done:
            for i in range(n):
                backup_target += self.gamma**i * rewards[t+i] + self.gamma**n * max(self.Q_sa[states[t+n],:])
        else:
            for i in range(n):
                backup_target += self.gamma**i * rewards[t+i]
        self.Q_sa[states[t], actions[t]] = self.Q_sa[states[t], actions[t]] + self.learning_rate * (backup_target - self.Q_sa[states[t], actions[t]])
        pass

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    i = 0
    # TO DO: Write your n-step Q-learning algorithm here!
    while i <= n_timesteps:
        s = env.reset()
        a_ep = []
        s_ep = [s]
        r_ep = []
        for t in range(max_episode_length):
            a = pi.select_action(s_ep[t], policy, epsilon, temp)
            s_next, r, done = env.step(a)
            a_ep.append(a)
            s_ep.append(s_next)
            r_ep.append(r)
            if done:
                break
        T_ep = t
        for t in range(T_ep):
            m = min(n,T_ep-t)
            pi.update(s_ep, a_ep, r_ep, done, m, t)
            if i%eval_interval == 0:
                eval_timesteps.append(i)
                eval_returns.append(pi.evaluate(eval_env))
                if plot:
                    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution
            i += 1
        
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    eval_returns, eval_timesteps = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print(eval_returns, eval_timesteps)
    
if __name__ == '__main__':
    test()
