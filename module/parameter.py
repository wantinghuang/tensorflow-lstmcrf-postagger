# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:38:03 2018

@author: wanting_huang
"""

class Config(object):
    def __init__(self):
        self.n_inputs = 100 # embedding_size
        self.n_classes = 472
        self.n_hidden_units = 128        # neurons in hidden layer
        self.max_step = 105 # max_time_step

class TrainConfig(object):
    def __init__(self):
        self.initial_learning_rate = 0.001
        self.batch_size = 128
        self.num_epoch = 35  
        self.lr_eps = 1e-2
        self.lr_decay_rate = 0.7
        self.loss_eps = 1e-3