# coding=utf-8
"""
The test criteria based on DeepGauge
"""
from keras.models import Model
from copy import deepcopy
import numpy as np
import pickle
import os


class DeepGauge(object):
    def __init__(self, model, model_name, exclude_layer, benchmark=None, k=1000):
        self.model = model
        self.model_name = model_name  # the model's name that uses to create cache or find cache
        self.benchmark_data = benchmark
        self.k = k

        # the layers that are needed to compute coverage
        self.layer_to_compute = [layer for layer in self.model.layers
                                 if all(ex not in layer.name for ex in exclude_layer)]

        self.knc_cov_dict = {}  # k-multi section coverage
        self.nbc_cov_dict = {}  # neuron boundary coverage: [..., 0]:lower; [..., 1]: upper
        self.layer_bounds = {}  # neuron output boundary in each layer
        self.layer_bounds_bin = {} # the section of each neuron in each layer
        self.intermediate_model = None # the model to get intermediate layer's output
        self.init_bounds()

    def init_bounds(self):
        print 'init'
        for layer in self.layer_to_compute:
            if os.path.exists(
                    '/home/test/program/self-driving/testing/cache/{model_name}/train_outputs/'
                    '{layer_name}_bounds.pkl'.format(model_name=self.model_name, layer_name=layer.name)):
                # judge whether there is a cache
                with open(
                        '/home/test/program/self-driving/testing/cache/{model_name}/train_outputs/'
                        '{layer_name}_bounds.pkl'.format(model_name=self.model_name, layer_name=layer.name), 'rb') as f:
                    (low_bound, high_bound) = pickle.load(f)
                    self.layer_bounds[layer.name] = (low_bound, high_bound)
                    self.layer_bounds_bin[layer.name] = [np.linspace(low_bound[i], high_bound[i], self.k + 1)
                                                         for i in range(high_bound.shape[0])]
                self.knc_cov_dict[layer.name] = np.zeros((layer.output.get_shape()[-1], self.k), dtype='bool')
                self.nbc_cov_dict[layer.name] = np.zeros((layer.output.get_shape()[-1], 2), dtype='bool')
                outputs_layer = [layer.output for layer in self.layer_to_compute]
                outputs_layer.append(self.model.layers[-1].output)
                self.intermediate_model = Model(input=self.model.input, output=outputs_layer)
                with open(
                        '/home/test/program/self-driving/testing/cache/{model_name}/train_outputs/'
                        'layer_bounds.pkl'.format(model_name=self.model_name, layer_name=layer.name), 'wb') as f:
                    pickle.dump(self.layer_bounds, f, pickle.HIGHEST_PROTOCOL)
                continue

        intermediate_model = Model(input=self.model.input, output=[layer.output for layer in self.layer_to_compute])
        intermediate_outputs = intermediate_model.predict(self.benchmark_data)
        for i, layer_outputs in enumerate(intermediate_outputs):
            layer = self.layer_to_compute[i]
            mean_layer_outputs = np.asarray([np.mean(output[..., c]) for output in layer_outputs
                                   for c in range(output.shape[-1])]).reshape(layer_outputs.shape[0], -1)
            high_bound = np.max(mean_layer_outputs, axis=0)
            low_bound = np.min(mean_layer_outputs, axis=0)
            with open(
                    '/home/test/program/self-driving/testing/cache/{model_name}/train_outputs/'
                    '{layer_name}_bounds.pkl'.format(model_name=self.model_name, layer_name=layer.name), 'wb') as f:
                bounds = (low_bound, high_bound)
                pickle.dump(bounds, f, pickle.HIGHEST_PROTOCOL)
            self.layer_bounds[layer.name] = (low_bound, high_bound)
            self.layer_bounds_bin[layer.name] = [np.linspace(low_bound[i], high_bound[i], self.k + 1)
                                                 for i in range(high_bound.shape[0])]
            self.knc_cov_dict[layer.name] = np.zeros((layer.output.get_shape()[-1], self.k), dtype='bool')
            self.nbc_cov_dict[layer.name] = np.zeros((layer.output.get_shape()[-1], 2), dtype='bool')
            self.intermediate_model = intermediate_model

    def init_knc_coverage(self, x):
        """
        Init K-Multi Section Coverage based on the existing data
        :param x:
        :return:
        """
        if os.path.exists('/home/test/program/self-driving/testing/cache/{model_name}/test_outputs/'
                          'knc_coverage.pkl'.format(model_name=self.model_name)):
            with open('/home/test/program/self-driving/testing/cache/{model_name}/test_outputs/'
                      'knc_coverage.pkl'.format(model_name=self.model_name), 'rb') as f:
                self.knc_cov_dict = pickle.load(f)
        else:
            if x.ndim < 4:
                x = x[np.newaxis, ...]
            intermediate_outputs = self.intermediate_model.predict(x)
            for i, layer_outputs in enumerate(intermediate_outputs):
                layer = self.layer_to_compute[i]
                bins = self.layer_bounds_bin[layer.name]
                for output in layer_outputs:
                    for id_neuron in range(output.shape[-1]):
                        _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
                        if _bin >= self.k or self.knc_cov_dict[layer.name][id_neuron][_bin]:
                            continue
                        self.knc_cov_dict[layer.name][id_neuron][_bin] = True
            with open('/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/knc_coverage.pkl', 'wb') as f:
                pickle.dump(self.knc_cov_dict, f, pickle.HIGHEST_PROTOCOL)

    def init_nbc_coverage(self, x):
        """
        Init Strong Neuron Boundary Coverage based on the existing data
        :param x:
        :return:
        """
        if os.path.exists('/home/test/program/self-driving/testing/cache/{model_name}/test_outputs/'
                          'nbc_coverage.pkl'.format(model_name=self.model_name)):
            with open('/home/test/program/self-driving/testing/cache/{model_name}/test_outputs/'
                      'nbc_coverage.pkl'.format(model_name=self.model_name),'rb') as f:
                self.nbc_cov_dict = pickle.load(f)
        else:
            if x.ndim < 4:
                x = x[np.newaxis, ...]
            intermediate_outputs = self.intermediate_model.predict(x)
            for i, layer_outputs in enumerate(intermediate_outputs):
                layer = self.layer_to_compute[i]
                (low_bound, high_bound) = self.layer_bounds[layer.name]
                for output in layer_outputs:
                    for id_neuron in range(output.shape[-1]):
                        val = np.mean(output[..., id_neuron])
                        if val < low_bound[id_neuron] and not self.nbc_cov_dict[layer.name][id_neuron][0]:
                            self.nbc_cov_dict[layer.name][id_neuron][0] = True
                        elif val > high_bound[id_neuron] and not self.nbc_cov_dict[layer.name][id_neuron][1]:
                            self.nbc_cov_dict[layer.name][id_neuron][1] = True
                        else:
                            continue
            with open('/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/nbc_coverage.pkl', 'wb') as f:
                pickle.dump(self.nbc_cov_dict, f, pickle.HIGHEST_PROTOCOL)

    def get_new_covered_knc_sections_by_outputs(self, outputs):
        """
        Calculate the new covered sections on the KNC criterion, for the given outputs
        :param outputs:  the intermediate layer's outputs
        :param knc_dict: the tmp knc coverage graph
        :return:
        """
        new_covered_sections = 0
        cov_dict = deepcopy(self.knc_cov_dict)
        for i, layer_outputs in enumerate(outputs):
            layer = self.layer_to_compute[i]
            bins = self.layer_bounds_bin[layer.name]
            for output in layer_outputs:
                for id_neuron in range(output.shape[-1]):
                    _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
                    if _bin >= self.k or cov_dict[layer.name][id_neuron][_bin]:
                        continue
                    new_covered_sections = new_covered_sections + 1
                    cov_dict[layer.name][id_neuron][_bin] = True
        del outputs
        del cov_dict
        return new_covered_sections

    def get_new_covered_knc_sections(self, x):
        """
        Calculate the number of new covered sections on the KNC criterion, for the given inputs
        :param x:
        :return:
        """
        if x.ndim < 4:
            x = x[np.newaxis, ...]
        new_covered_sections = 0
        intermediate_outputs = self.intermediate_model.predict(x)
        for i, layer_outputs in enumerate(intermediate_outputs):
            layer = self.layer_to_compute[i]
            bins = self.layer_bounds_bin[layer.name]
            for output in layer_outputs:
                for id_neuron in range(output.shape[-1]):
                    _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
                    if _bin >= self.k or self.knc_cov_dict[layer.name][id_neuron][_bin]:
                        continue
                    new_covered_sections = new_covered_sections + 1
        return new_covered_sections

    def get_new_covered_knc_sections_without_duplicate(self, x):
        if x.ndim < 4:
            x = x[np.newaxis, ...]
        new_covered_sections = 0
        cov_dict = deepcopy(self.knc_cov_dict)
        intermediate_outputs = self.intermediate_model.predict(x)
        for i, layer_outputs in enumerate(intermediate_outputs):
            layer = self.layer_to_compute[i]
            bins = self.layer_bounds_bin[layer.name]
            for output in layer_outputs:
                for id_neuron in range(output.shape[-1]):
                    _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
                    if _bin >= self.k or cov_dict[layer.name][id_neuron][_bin]:
                        continue
                    new_covered_sections = new_covered_sections + 1
                    cov_dict[layer.name][id_neuron][_bin] = True
        return new_covered_sections

    def update_knc(self, x):
        """
        Update K-Multi Section Neuron Coverage based on the given inputs
        :param x:
        :return:
        """
        if x.ndim < 4:
            x = x[np.newaxis, ...]
        intermediate_outputs = self.intermediate_model.predict(x)
        for i, layer_outputs in enumerate(intermediate_outputs):
            layer = self.layer_to_compute[i]
            bins = self.layer_bounds_bin[layer.name]
            for output in layer_outputs:
                for id_neuron in range(output.shape[-1]):
                    _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
                    if _bin >= self.k or self.knc_cov_dict[layer.name][id_neuron][_bin]:
                        continue
                    self.knc_cov_dict[layer.name][id_neuron][_bin] = True

    def current_knc_coverage(self):
        """
        Calculate the current K-Multi Section Neuron Coverage
        :return:
        """
        covered = 0
        total = 0
        for layer in self.layer_to_compute:
            covered = covered + np.count_nonzero(self.knc_cov_dict[layer.name])
            total = total + np.size(self.knc_cov_dict[layer.name])
        return covered / float(total)

    def calculate_uncovered_sections(self):
        """
        Calculate the number of uncovered sections on the KNC criterion
        :return:
        """
        count = 0
        for layer in self.layer_to_compute:
            count = count + np.count_nonzero(self.knc_cov_dict[layer.name] == 0)
        return count

    def get_new_covered_nbc_sections(self, x):
        """
        Calculate the number of new covered section on the NBC criterion, for the given inputs
        :param x:
        :return:
        """
        if x.ndim < 4:
            x = x[np.newaxis, ...]
        intermediate_outputs = self.intermediate_model.predict(x)
        new_covered_sections = 0
        for i, layer_outputs in enumerate(intermediate_outputs):
            layer = self.layer_to_compute[i]
            (low_bound, high_bound) = self.layer_bounds[layer.name]
            for output in layer_outputs:
                for id_neuron in range(output.shape[-1]):
                    val = np.mean(output[..., id_neuron])
                    if val < low_bound[id_neuron] and not self.nbc_cov_dict[layer.name][id_neuron][0]:
                        new_covered_sections = new_covered_sections + 1
                    elif val > high_bound[id_neuron] and not self.nbc_cov_dict[layer.name][id_neuron][1]:
                        new_covered_sections = new_covered_sections + 1
                    else:
                        continue
        return new_covered_sections

    def update_nbc(self, x):
        """
        Update Neuron Boundary Coverage based on the given inputs
        :param x:
        :return:
        """
        if x.ndim < 4:
            x = x[np.newaxis, ...]
        intermediate_outputs = self.intermediate_model.predict(x)
        for i, layer_outputs in enumerate(intermediate_outputs):
            layer = self.layer_to_compute[i]
            (low_bound, high_bound) = self.layer_bounds[layer.name]
            for output in layer_outputs:
                for id_neuron in range(output.shape[-1]):
                    val = np.mean(output[..., id_neuron])
                    if val < low_bound[id_neuron] and not self.nbc_cov_dict[layer.name][id_neuron][0]:
                        self.nbc_cov_dict[layer.name][id_neuron][0] = True
                    elif val > high_bound[id_neuron] and not self.nbc_cov_dict[layer.name][id_neuron][1]:
                        self.nbc_cov_dict[layer.name][id_neuron][1] = True
                    else:
                        continue

    def current_nbc_coverage(self):
        """
        Calculate the current Neuron Boundary Coverage
        :return:
        """
        covered = 0
        total = 0
        for layer in self.layer_to_compute:
            covered = covered + np.count_nonzero(self.nbc_cov_dict[layer.name])
            total = total + np.size(self.nbc_cov_dict[layer.name])
        return covered / float(total)

    def calculate_nb_neurons(self):
        """
        Calculate the amount of effective neurons in the model
        :return:
        """
        count = 0
        for layer in self.layer_to_compute:
            count = count + int(layer.output.get_shape()[-1])
        return count