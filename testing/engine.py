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
            break
            # if self.count == 60:
            #     return self.best_individual
            # # parent = self.kill_bad(parent, kid)
        return self.best_individual

    def make_kid(self, parent):
        kid = parent + self.mut_strength * torch.randn(self.style_dim, 1, 1)
        return kid

    def kill_bad(self, kid):
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


class EAEngineV(object):
    def __init__(self, style_dim, fitness_func, logger):
        self.style_dim = style_dim

        self.fitness = fitness_func

        # self.mut_strength = 5.  # step size(dynamic mutation strength)

        self.mutation_rate = 1. / float(style_dim)

        self.best_fitness = 0.

        self.best_individual = None

        self.logger = logger

        self.count = 0

    def run(self, max_iter):
        self.best_individual = torch.randn(self.style_dim, 1, 1)
        self.best_fitness = self.fitness(self.best_individual)
        for iter in range(max_iter):
            self.logger.info("the {i}'s fitness is {v}".format(i=iter, v=self.best_fitness))
            kid = self.make_kid(self.best_individual)
            # self.logger.info("the parent is {}".format(self.best_individual))
            # self.logger.info("the kid is {}".format(kid))
            self.kill_bad(kid)
            if self.count == 60:
                return self.best_individual
            # parent = self.kill_bad(parent, kid)
        return self.best_individual

    def make_kid(self, parent):
        # kid = parent + self.mut_strength * torch.randn(self.style_dim, 1, 1)
        kid = deepcopy(parent)
        _new = torch.randn(8, 1, 1)
        for i in range(self.style_dim):
            if np.random.rand() < self.mutation_rate:
                kid[i, 0, 0] = _new[i, 0, 0]
        return kid

    def kill_bad(self, kid):
        kid_fitness = self.fitness(kid)
        if self.best_fitness < kid_fitness:
            self.best_individual = kid
            self.best_fitness = kid_fitness
            self.count = 0
        else:
            self.count += 1
        # p_target = 1 / 5
        # if self.best_fitness < kid_fitness:
        #     self.best_individual = kid
        #     self.best_fitness = kid_fitness
        #     ps = 1.
        #     self.count = 0
        # else:
        #     ps = 0.
        #     self.count += 1
        # self.mut_strength *= np.exp(1 / np.sqrt(self.style_dim + 1) * (ps - p_target) / (1 - p_target))

    # def kill_bad(self, parent, kid):
    #     # parent_fitness = self.fitness(parent)
    #     kid_fitness = self.fitness(kid)
    #
    #     p_target = 1 / 5
    #     if parent_fitness < kid_fitness:
    #         parent = kid
    #         ps = 1.
    #         self.best_fitness = kid_fitness
    #     else:
    #         ps = 0.
    #     self.mut_strength *= np.exp(1 / np.sqrt(self.style_dim + 1) * (ps - p_target) / (1 - p_target))
    #     return parent


class RandomSearch(object):
    def __init__(self, style_dim, fitness_func, logger):
        self.style_dim = style_dim

        self.fitness = fitness_func

        # self.mut_strength = 5.  # step size(dynamic mutation strength)

        # self.mutation_rate = 1 / style_dim

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
            # parent = self.kill_bad(parent, kid)
        return self.best_individual

    # def make_kid(self, parent):
    #     # kid = parent + self.mut_strength * torch.randn(self.style_dim, 1, 1)
    #     return torch.randn(self.style_dim, 1, 1)

    def kill_bad(self, kid):
        kid_fitness = self.fitness(kid)
        if self.best_fitness < kid_fitness:
            self.best_individual = kid
            self.best_fitness = kid_fitness
            self.count = 0
        else:
            self.count += 1


class GAEngine(object):
    def __init__(self, style_dim, fitness_func, nb_population, pc, pm):
        self.style_dim = style_dim
        self.nb_population = nb_population
        self.pc = pc
        self.pm = pm

        self.fitness_func = fitness_func

    def run(self, max_iter):
        population = [torch.randn(self.style_dim, 1, 1) for _ in range(self.nb_population)]
        fitness = self.calculate_fitness(population)
        for i in range(max_iter):
            _best, _ = self.best(population, fitness)
            #print _best_val

            local_pop = []

            # generate offsprings
            for _ in range(self.nb_population // 2):
                parents = self.select(population, fitness)
                children = self.crossover(*parents)
                children = [self.mutate(child) for child in children]
                local_pop.extend(children)

            local_pop[0] = _best.clone()
            del _best

            population = local_pop
            del local_pop
            fitness = self.calculate_fitness(population)

        best, _ = self.best(population, fitness)
        return best

    def calculate_fitness(self, population):
        fitness = [self.fitness_func(individual) for individual in population]
        return np.asarray(fitness)

    def crossover(self, father, mother):
        # uniform cross
        offspring1 = father.clone()
        del father
        offspring2 = mother.clone()
        del mother

        do_cross = True if random() <= self.pc else False
        if not do_cross:
            return offspring1, offspring2

        for i in range(offspring1.shape[0]):
            do_exchange = True if random() <= self.pc else False
            if do_exchange:
                offspring1[i], offspring2[i] = offspring2[0].item(), offspring1[0].item()

        return offspring1, offspring2

    def mutate(self, individual):
        offspring = individual.clone()
        del individual
        do_mutation = True if random() <= self.pm else False
        if not do_mutation:
            return offspring
        for i in range(offspring.shape[0]):
            do_mutation = True if random() <= self.pm else False
            if do_mutation:
                offspring[i] = torch.randn(1).item()
        return offspring

    def select(self, population, fitness):
        # roulette-wheel selection
        # nb_pop = len(population)
        # idx_1 = np.random.choice(np.arange(nb_pop), size=nb_pop, replace=True,
        #                          p=fitness / np.sum(fitness))
        idx_1 = self.stochastic_accept(fitness)
        father = population[idx_1]
        idx_2 = (idx_1 + 1) % len(population)
        mother = population[idx_2]
        return father, mother

    @staticmethod
    def stochastic_accept(fitness):
        N = fitness.shape[0]
        max_fit = np.max(fitness)
        while True:
            idx = int(N * random())
            if random() <= fitness[idx] / max_fit:
                return idx

    @staticmethod
    def best(population, fitness):
        # find the best individual
        idx = np.argmax(fitness)
        return population[idx], fitness[idx]
