# coding=utf-8
from keras.models import Model
import numpy as np
import pickle


def calculate_uncovered_sections(cov_dict):
    """
    Calculate the number of uncovered sections
    """
    return np.sum([np.count_nonzero(v == 0) for v in cov_dict.values()])


def get_new_covered_knc_sections(intermediate_layer_outputs, layer_bounds_bins, nb_part, cov_dict):
    """
    Calculate the number of sections that are covered by given outputs, based on KNC criterion, without duplicate
    :param intermediate_layer_outputs: the given outputs
    :param layer_bounds_bins: the dictionary which save the bin of each neuron in model
    :param nb_part: the value of K, K is the number of neuron's sections
    :param cov_dict: the graph which record the coverage information about KNC
    """
    new_covered_sections = 0
    for i, layer in enumerate(cov_dict.keys()):
        output = intermediate_layer_outputs[i][0]
        bins = layer_bounds_bins[layer]
        for id_neuron in range(output.shape[-1]):
            _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
            if _bin >= nb_part or cov_dict[layer][id_neuron][_bin]:
                continue
            new_covered_sections = new_covered_sections + 1
            cov_dict[layer][id_neuron][_bin] = True
    return new_covered_sections


def update_knc(intermediate_layer_outputs, layer_bounds_bins, nb_part, cov_dict):
    """
    Update the coverage information based on KNC criterion
    :param intermediate_layer_outputs: the given outputs
    :param layer_bounds_bins: the dictionary which save the bin of each neuron in model
    :param nb_part: the value of K, K is the number of neuron's sections
    :param cov_dict: the graph which record the coverage information about KNC
    :return:
    """
    for i, layer in enumerate(cov_dict.keys()):
        output = intermediate_layer_outputs[i][0]
        bins = layer_bounds_bins[layer]
        for id_neuron in range(output.shape[-1]):
            _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
            if _bin >= nb_part or cov_dict[layer][id_neuron][_bin]:
                continue
            cov_dict[layer][id_neuron][_bin] = True


def get_new_covered_nbc_sections(intermediate_layer_outputs, layer_bounds, cov_dict):
    """
    Calculate the number of sections that are covered by given outputs, based on NBC criterion, without duplicate
    :param intermediate_layer_outputs: the given outputs
    :param layer_bounds: the dictionary which save the boundary of each neuron in model
    :param cov_dict: the graph which record the coverage information about NBC
    """
    new_covered_sections = 0
    for i, layer in enumerate(cov_dict.keys()):
        output = intermediate_layer_outputs[i][0]
        print output.shape
        (low_bound, high_bound) = layer_bounds[layer]
        for id_neuron in range(output.shape[-1]):
            val = np.mean(output[..., id_neuron])
            if val < low_bound[id_neuron] and not cov_dict[layer][id_neuron][0]:
                new_covered_sections = new_covered_sections + 1
                cov_dict[layer][id_neuron][0] = True
            elif val > high_bound[id_neuron] and not cov_dict[layer][id_neuron][1]:
                new_covered_sections = new_covered_sections + 1
                cov_dict[layer][id_neuron][1] = True
            else:
                continue
    return new_covered_sections


def update_nbc(intermediate_layer_outputs, layer_bounds, cov_dict):
    """
    Update the coverage information based on NBC criterion
    :param intermediate_layer_outputs: the given outputs
    :param layer_bounds: the dictionary which save the boundary of each neuron in model
    :param cov_dict: the graph which record the coverage information about NBC
    """
    for i, layer in enumerate(cov_dict.keys()):
        output = intermediate_layer_outputs[i][0]
        (low_bound, high_bound) = layer_bounds[layer]
        for id_neuron in range(output.shape[-1]):
            val = np.mean(output[..., id_neuron])
            if val < low_bound[id_neuron] and not cov_dict[layer][id_neuron][0]:
                cov_dict[layer][id_neuron][0] = True
            elif val > high_bound[id_neuron] and not cov_dict[layer][id_neuron][1]:
                cov_dict[layer][id_neuron][1] = True
            else:
                continue


def get_coverage(cov_dict):
    """
    Calculate the current coverage based on the given coverage graph
    :return:
    """
    covered = 0
    total = 0
    for v in cov_dict.values():
        covered = covered + np.count_nonzero(v)
        total = total + np.size(v)
    return covered / float(total)


def init_gauge_without_cache(model, model_name, layer_to_compute, nb_part, x_train, x_test):
    """
    Init DeepGauge without cache
    :param model: the target model
    :param model_name: the target model's name
    :param layer_to_compute: The layers in the target model that need to compute
    :param nb_part: the value of K, K is the number of neuron's sections
    :param x_train: the train data of the target model
    :param x_test: the original test data of the target model
    :return:
    """
    layer_bounds_bin = {}
    layer_bounds = {}
    knc_cov_dict = {}
    nbc_cov_dict = {}

    intermediate_model = Model(input=model.input, output=[layer.output for layer in layer_to_compute])
    # init boundary
    intermediate_outputs = intermediate_model.predict(x_train)
    for i, layer_outputs in enumerate(intermediate_outputs):
        layer = layer_to_compute[i]
        mean_layer_outputs = np.asarray([np.mean(output[..., c]) for output in layer_outputs
                                         for c in range(output.shape[-1])]).reshape(layer_outputs.shape[0], -1)
        high_bound = np.max(mean_layer_outputs, axis=0)
        low_bound = np.min(mean_layer_outputs, axis=0)
        layer_bounds_bin[layer.name] = [np.linspace(low_bound[i], high_bound[i], nb_part + 1)
                                        for i in range(high_bound.shape[0])]
        layer_bounds[layer.name] = (low_bound, high_bound)
        knc_cov_dict[layer.name] = np.zeros((layer.output.get_shape()[-1], nb_part), dtype='bool')
        nbc_cov_dict[layer.name] = np.zeros((layer.output.get_shape()[-1], 2), dtype='bool')
    with open(
            '/home/test/program/self-driving/testing/cache/{model_name}/train_outputs/'
            'layer_bounds.pkl'.format(model_name=model_name), 'wb') as f:
        pickle.dump(layer_bounds, f, pickle.HIGHEST_PROTOCOL)
    with open(
            '/home/test/program/self-driving/testing/cache/{model_name}/train_outputs/'
            'layer_bounds_bin.pkl'.format(model_name=model_name), 'wb') as f:
        pickle.dump(layer_bounds_bin, f, pickle.HIGHEST_PROTOCOL)
    del intermediate_outputs
    # init KNC
    intermediate_outputs = intermediate_model.predict(x_test)
    for i, layer_outputs in enumerate(intermediate_outputs):
        layer = layer_to_compute[i]
        bins = layer_bounds_bin[layer.name]
        for output in layer_outputs:
            for id_neuron in range(output.shape[-1]):
                _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
                if _bin >= nb_part or knc_cov_dict[layer.name][id_neuron][_bin]:
                    continue
                knc_cov_dict[layer.name][id_neuron][_bin] = True
    with open(
            '/home/test/program/self-driving/testing/cache/{model_name}/'
            'test_outputs/knc_coverage.pkl'.format(model_name=model_name), 'wb') as f:
        pickle.dump(knc_cov_dict, f, pickle.HIGHEST_PROTOCOL)
    # init NBC
    for i, layer_outputs in enumerate(intermediate_outputs):
        layer = layer_to_compute[i]
        (low_bound, high_bound) = layer_bounds[layer.name]
        for output in layer_outputs:
            for id_neuron in range(output.shape[-1]):
                val = np.mean(output[..., id_neuron])
                if val < low_bound[id_neuron] and not nbc_cov_dict[layer.name][id_neuron][0]:
                    nbc_cov_dict[layer.name][id_neuron][0] = True
                elif val > high_bound[id_neuron] and not nbc_cov_dict[layer.name][id_neuron][1]:
                    nbc_cov_dict[layer.name][id_neuron][1] = True
                else:
                    continue
    with open(
            '/home/test/program/self-driving/testing/cache/{model_name}/'
            'test_outputs/nbc_coverage.pkl'.format(model_name=model_name), 'wb') as f:
        pickle.dump(nbc_cov_dict, f, pickle.HIGHEST_PROTOCOL)


def init_gauge_with_gauge(model_name, flag=0):
    """
    Init DeepGauge with cache
    :param model_name: the target model's name
    :param flag: when flag=1, init information about KNC, when flag=0, init information about NBC
    :return:
    """
    if flag == 0:
        with open(
                '/home/test/program/self-driving/testing/cache/{}/'
                'train_outputs/layer_bounds.pkl'.format(model_name), 'rb') as f:
            layer_bounds = pickle.load(f)
        with open('/home/test/program/self-driving/testing/cache/{}/'
                  'test_outputs/nbc_coverage.pkl'.format(model_name), 'rb') as f:
            nbc_cov_dict = pickle.load(f)
        return layer_bounds, nbc_cov_dict
    elif flag == 1:
        with open('/home/test/program/self-driving/testing/cache/Chauffeur/train_outputs/layer_bounds_bin.pkl', 'rb') as f:
            layer_bounds_bins = pickle.load(f)
        with open('/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/knc_coverage_cache.pkl', 'rb') as f:
            knc_cov_dict = pickle.load(f)
        return layer_bounds_bins, knc_cov_dict
    else:
        raise ValueError("Invalid Initial Model")
