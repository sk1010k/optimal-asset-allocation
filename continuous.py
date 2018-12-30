#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import envs
import agents

matplotlib.style.use('ggplot')
Log = namedtuple('Log', ['states', 'actions', 'rewards'])


def make_epsilon_greedy_policy(agent, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(state):
        A = np.ones(nA, dtype=float) * epsilon / nA
        s = np.array(state).reshape(1,-1)
        q_values = agent.predict(s)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, agent, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, decay_exponent=0):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """ 
    
    log = Log([], [], [])
        
    # The policy we're following
    policy = make_epsilon_greedy_policy(
        agent, epsilon * epsilon_decay**decay_exponent, env.action_space.n)
    
    # Reset the environment and pick the first action
    state = env.reset()
    log.states.append(state)    
    
    # One step in the environment
    for t in itertools.count():
                    
        # Choose an action to take
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        log.actions.append(action)
        
        # Take a step
        next_state, reward, done, _ = env.step(action)
        log.rewards.append(reward)
        
        # TD Update
        s = np.array(state).reshape(1,-1)
        q_values_next = agent.predict(s)
        
        # Use this code for Q-Learning
        # Q-Value TD Target
        td_target = reward + discount_factor * np.max(q_values_next)
        
        # Update the function approximator using our target
        a = np.array(action).reshape(-1,1)
        y = np.array(td_target).reshape(-1,1)
        agent.update(s, a, y)
        # print(list(agent.model.parameters())[0][0][0]) # For debugging.

        if done:
            break
            
        state = next_state
        log.states.append(state)
    
    return log


if __name__ == '__main__':
    
    prices = pd.read_csv('./data/USD_JPY.csv').Price.values
    train_prices = prices[:2000]
    test_prices = prices[2000:]
    currency_config = envs.USDJPYConfig()

    # Training
    env = envs.TwoAssetsContinuousEnv(train_prices, currency_config)
    agent = agents.Agent(env)
    print('Training...')
    log = q_learning(env, agent)
    agent.save_model('models/model.pth')
    
    # Test
    agent.load_model('models/model.pth')
    env.reset_prices(test_prices)
    env.reset()
    print('Testing...')
    log = q_learning(env, agent)
    
    prices, capital, _ = list(zip(*log.states))
    plt.figure(figsize=(12,6))
    plt.subplot(311)
    plt.plot(prices)
    plt.subplot(312)
    plt.plot(np.array(log.rewards).cumsum())
    plt.subplot(313)
    plt.plot(log.actions)
