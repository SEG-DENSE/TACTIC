# coding=utf-8
import torch
import numpy as np
from random import random
from copy import deepcopy


class EAEngine(object):
    def __init__(self, style_dim, fitness_func, logger):
        self.style_dim = style_dim

        self.fitness = fitness_func

        self.mut_strength = 5.  # step size(dynamic mutation strength)

        self.best_fitness = 0.

        self.best_individual = None

        self.logger = logger

        self.count = 0

    def run(self, max_iter):
        self.best_individual = torch.randn(self.style_dim, 1, 1)
        self.logger.info("the initial style code is {}".format(self.best_individual))
        self.best_fitness = self.fitness(self.best_individual)
        for iter in range(max_iter):
            self.logger.info("the {i}'s fitness is {v}".format(i=iter, v=self.best_fitness))
            kid = self.make_kid(self.best_individual)
            self.logger.info("generate kid")
            self.kill_bad(kid)
            self.logger.info("finish selection")
            if self.count == 60:
                 return self.best_individual
        return self.best_individual

    def make_kid(self, parent):
        kid = parent + self.mut_strength * torch.randn(self.style_dim, 1, 1)
        return kid

    def kill_bad(self, kid):
        # This is a comment
        kid_fitness = self.fitness(kid)
        p_target = 1 / 5
        if self.best_fitness < kid_fitness:
            self.best_individual = kid
            self.best_fitness = kid_fitness
            ps = 1.
            self.count = 0
        else:
            ps = 0.
            self.count += 1
        self.mut_strength *= np.exp(1 / np.sqrt(self.style_dim + 1) * (ps - p_target) / (1 - p_target))


class RandomSearch(object):
    def __init__(self, style_dim, fitness_func, logger):
        self.style_dim = style_dim

        self.fitness = fitness_func

        self.best_fitness = 0.

        self.best_individual = None

        self.logger = logger

        self.count = 0

    def run(self, max_iter):
        self.best_individual = torch.randn(self.style_dim, 1, 1)
        self.logger.info("the initial style code is {}".format(self.best_individual))
        self.best_fitness = self.fitness(self.best_individual)
        for iter in range(max_iter):
            self.logger.info("the {i}'s fitness is {v}".format(i=iter, v=self.best_fitness))
            kid = torch.randn(self.style_dim, 1, 1)
            self.kill_bad(kid)
            if self.count == 60:
                return self.best_individual
        return self.best_individual

    def kill_bad(self, kid):
        kid_fitness = self.fitness(kid)
        if self.best_fitness < kid_fitness:
            self.best_individual = kid
            self.best_fitness = kid_fitness
            self.count = 0
        else:
            self.count += 1
