#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable


class MLP(nn.Module):
    
    def __init__(self, layers):
        """
        Args:
            layers: a list of the number of units in each layer. len(layers) >= 3.
        """
        
        super(MLP, self).__init__()
        
        if len(layers) <= 2:
            raise Exception('len(layers) must be >= 3')
            
        self.model = nn.Sequential()
        for i in range(len(layers)-2):
            self.model.add_module('fc{}'.format(i+1), nn.Linear(layers[i], layers[i+1]))
            self.model.add_module('relu{}'.format(i+1), nn.ReLU())
        self.model.add_module('fc{}'.format(i+2), nn.Linear(layers[i+1], layers[i+2]))
        
    def forward(self, x):
        return self.model(x)


class Agent:
    """Use neural networks for function approximator."""
    
    def __init__(self, env):
                
        # Can't set the first argment from env automatically rather than constant STATE_DIM ?
        self.model = MLP([len(env.observation_space.spaces), 5, env.action_space.n])
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
    
    def predict(self, s):
        """
        Predicts action values.

        Args:
          s (float np.ndarray): State input of shape [batch_size, STATE_DIM]

        Returns:
          NumPy array of shape [batch_size, env.action_space.n] containing the estimated 
          action values.
        """
        
        s = Variable(torch.from_numpy(s)).float()
        
        self.model.eval()
        output = self.model(s)
        return output.data.numpy()

    def update(self, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          s (float np.ndarray): State input of shape [batch_size, STATE_DIM]
          a (int np.ndarray): Chosen actions of shape [batch_size]
          y (float np.ndarray): Targets of shape  [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        
        s = Variable(torch.from_numpy(s)).float()
        a = Variable(torch.from_numpy(a)).long()
        y = Variable(torch.from_numpy(y)).float()
        
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(s)
        # Get the predictions for the chosen actions only.
        output = torch.gather(output, 1, a.view(-1,1))
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.data.numpy()

    def save_model(self, path):

        torch.save(self.model.state_dict(), path)

    def load_model(self, path):

        self.model.load_state_dict(torch.load(path))
        self.model.eval()
