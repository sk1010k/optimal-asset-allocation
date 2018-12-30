#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class USDJPYConfig:

    QUOTE_UNIT = 0.01 # JPY unit.
    QUOTE_N_DIGITS = 2
    QUOTE_MAX = 200.0
    QUOTE_MIN = 0.0
    QUOTE_N_BINS = int((QUOTE_MAX-QUOTE_MIN)/QUOTE_UNIT) + 1
    
    CAPITAL_UNIT = 0.01
    CAPITAL_N_DIGITS = 2
    CAPITAL_MAX = 10000.0 # Investment in jPY.
    CAPITAL_MIN = -10000.0 # Investment in USD.
    CAPITAL_N_BINS = int((CAPITAL_MAX-CAPITAL_MIN)/CAPITAL_UNIT) + 1
    
    SPREAD = 0.1
    
    IN_QUOTE = 0 # Investment in JPY.
    IN_BASE = 1 # Investment in USD. 


class TwoAssetsDiscreteEnv(gym.Env):
    """
    Implementation of Optimal Asset Allocation using Adaptive Dynamic Programming by Ralph Neuneier (1996).
    State is discrete. No function approximation.
    """

    def __init__(self, prices, currency_config, capital=1000):
        self.config = currency_config
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.config.QUOTE_N_BINS), # Space of quote currency.
            spaces.Discrete(self.config.CAPITAL_N_BINS), # Space of capital
            spaces.Discrete(2))) # Current investment; 0 for quote and 1 for base currency.
        self._seed()
        self.prices = prices
        self.init_capital = capital
        self._reset()
        self.nA = 2

    def step(self, action):
        return self._step(action)

    def reset(self):
        return self._reset()
        
    def _discretize_capital(self, capital):
        discrete_capital = round(float(capital), self.config.CAPITAL_N_DIGITS)
        capital_state = (discrete_capital + -self.config.CAPITAL_MIN) * \
                         10**self.config.CAPITAL_N_DIGITS
        return int(capital_state)
    
    def _discretize_price(self, price):
        discrete_price = round(float(price), self.config.QUOTE_N_DIGITS)
        price_state = max(self.config.QUOTE_MIN,
            (discrete_price - self.config.QUOTE_MIN) * 10**self.config.QUOTE_N_DIGITS)
        return int(price_state)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        self.price_step += 1
        self.next_price = self.prices[self.price_step]
        if action == self.config.IN_QUOTE:  # Change or keep the portfolio in quote currency.
            done = False
            reward = 0
        else:  # Change or keep the portfolio in base currency.
            if self.capital > 0:
                cost = self._calc_cost()
                reward = (self.next_price/self.current_price)*(self.capital-cost) - self.capital
                self.capital += reward
                self.capital *= -1 # Change the investment in base currency.
            elif self.capital < 0:
                # Different from the original paper; reward is always positive if it's preferrable
                reward = - (self.next_price/self.current_price - 1) * self.capital 
                self.capital -= reward
            else:
                raise ValueError('No capital!')
        self.current_price = self.next_price
        self.price_state = self._discretize_price(self.current_price)
        self.capital_state = self._discretize_capital(self.capital)
        done = self._check_if_done()
        self.state = (self.price_state, self.capital_state, action)
        return self.state, reward, done, {}
    
    def _calc_cost(self):
        cost = self.config.SPREAD * self.capital / self.current_price
        return cost
    
    def _check_if_done(self):
        if self.price_step == len(self.prices)-1 or self.capital >= self.config.CAPITAL_MAX or self.capital <= self.config.CAPITAL_MIN or self.capital == 0:
            done = True
        else:
            done = False
        return done
    
    def reset_prices(self, prices):
        self.prices = prices
    
    def _reset(self):
        self.price_step = 0
        self.current_price = self.prices[self.price_step]
        self.price_state = self._discretize_price(self.current_price)
        self.capital = self.init_capital
        self.capital_state = self._discretize_capital(self.capital)
        return (self.price_state, self.capital_state, 0) # Initially have investment in quote currency.
    
    

class TwoAssetsContinuousEnv(gym.Env):
    """Implementation of Optimal Asset Allocation using Adaptive Dynamic Programming by Ralph Neuneier (1996).
       State is continuous. Use function approximation."""

    def __init__(self, prices, currency_config, capital=1000):
        self.config = currency_config
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Box(self.config.QUOTE_MIN, self.config.QUOTE_MAX, shape=(1,), dtype=np.float32), # Space of quote currency.
            spaces.Box(self.config.CAPITAL_MIN, self.config.CAPITAL_MAX, shape=(1,), dtype=np.float32), # Space of capital.
            spaces.Discrete(2))) # Current investment; 0 for quote and 1 for base currency.
        self._seed()
        self.prices = prices
        self.init_capital = capital
        self._reset()
        self.nA = 2

    def step(self, action):
        return self._step(action)

    def reset(self):
        return self._reset()
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        self.price_step += 1
        self.next_price = self.prices[self.price_step]
        if action == self.config.IN_QUOTE: # Change or keep the portfolio in quote currency.
            reward = 0
        else:  # Change or keep the portfolio in base currency.
            if self.capital > 0:
                cost = self._calc_cost()
                reward = (self.next_price/self.current_price)*(self.capital-cost) - self.capital
                self.capital += reward
                self.capital *= -1 # Change the investment in base currency.
            elif self.capital < 0:
                # Different from the original paper; reward is always positive if it's preferrable
                reward = - (self.next_price/self.current_price - 1) * self.capital 
                self.capital -= reward
            else:
                raise ValueError('No capital!')
        done = self._check_if_done()
        self.current_price = self.next_price
        self.state = (self.current_price, self.capital, action)
        return self.state, reward, done, {}
    
    def _calc_cost(self):
        cost = self.config.SPREAD * self.capital / self.current_price
        return cost
    
    def _check_if_done(self):
        if self.price_step == len(self.prices)-1 or self.capital >= self.config.CAPITAL_MAX or self.capital <= self.config.CAPITAL_MIN or self.capital == 0:
            done = True
        else:
            done = False
        return done
    
    def reset_prices(self, prices):
        self.prices = prices
    
    def _reset(self):
        self.price_step = 0
        self.current_price = self.prices[self.price_step]
        self.capital = self.init_capital
        return (self.current_price, self.capital, 0) # 0 means we Initially have investment in quote currency.
