# coding=utf-8
from __future__ import print_function
from keras.models import model_from_json
from keras import backend as K
from keras.models import Model
from munit.munit import MUNIT
from munit.utils import get_config
from testing.chauffeur.chauffeur_data_utils import process_input
from testing.engine import RandomSearch, EAEngine
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from collections import deque
from copy import deepcopy
import os
import logging
import torch
import cv2
import pickle
import numpy as np

####################
# parameters
####################
# the image path
train_image_paths = '/home/test/program/self-driving/dataset/train/center/'
test_image_paths = '/home/test/program/self-driving/dataset/test/center/'
# the cache of original images steering angles
original_preds_cache_path = '/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/steering_angles.pkl'
# munit model path
# config_path = '/home/test/program/self-driving/munit/configs/night.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/night/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/rainy.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/rainy/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/sunny.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/sunny/gen_01250000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/snow.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/snow/gen_01000000.pt'
config_path = '/home/test/program/self-driving/munit/configs/snow_night.yaml'
checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/snow_night/gen_01000000.pt'
# the self-driving system's weight file
cnn_json_path = '/home/test/program/self-driving/models/chauffeur/pretrained/cnn.json'
cnn_weights_path = '/home/test/program/self-driving/models/chauffeur/pretrained/cnn.weights'
lstm_json_path = '/home/test/program/self-driving/models/chauffeur/pretrained/lstm.json'
lstm_weights_path = '/home/test/program/self-driving/models/chauffeur/pretrained/lstm.weights'

###################
# set logger
###################
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("./logger/knc_snow_night_ES_time_cost.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


##################
# create self-driving model
##################
K.set_learning_phase(0)


def load_from_json(json_path, weights_path):
    model = model_from_json(open(json_path, 'r').read())
    model.load_weights(weights_path)
    return model


def load_encoder(cnn_json, cnn_weights):
    model = load_from_json(cnn_json, cnn_weights)
    model.load_weights(cnn_weights)

    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    return model


Chauffeur_Encoder = load_encoder(cnn_json_path, cnn_weights_path)
Chauffeur_LSTM = load_from_json(lstm_json_path, lstm_weights_path)

Chauffeur_Encoder.summary()

# two important parameters for Chauffeur: hardcoded from final submission model
Chauffeur_scale = 16.
Chauffeur_timesteps = 100
#####################
# init coverage handler
#####################
layer_to_compute = [layer for layer in Chauffeur_Encoder.layers
                    if all(ex not in layer.name for ex in ['pool', 'fc', 'flatten', 'input'])]
nb_part = 1000  # the number of intervals that KNC needs to divide
# init coverage dict
with open('/home/test/program/self-driving/testing/cache/Chauffeur/train_outputs/layer_bounds_bin.pkl', 'rb') as f:
    layer_bounds_bins = pickle.load(f)
with open('/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/knc_coverage.pkl', 'rb') as f:
    knc_cov_dict = pickle.load(f)
# with open('/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/knc_coverage_cache_rainy_random_1.pkl', 'rb') as f:
#     knc_cov_dict = pickle.load(f)

outputs_layer = [layer.output for layer in layer_to_compute]
outputs_layer.append(Chauffeur_Encoder.layers[-1].output)
intermediate_model = Model(input=Chauffeur_Encoder.input, output=outputs_layer)
del Chauffeur_Encoder

nb_neurons = np.sum([int(layer.output.get_shape()[-1]) for layer in layer_to_compute])

#####################
# build MUNIT model
#####################
config = get_config(config_path)

munit = MUNIT(config)

try:
    state_dict = torch.load(checkpoint_path)
    munit.gen_a.load_state_dict(state_dict['a'])
    munit.gen_b.load_state_dict(state_dict['b'])
except Exception:
    raise RuntimeError('load model failed')

munit.cuda()
new_size = config['new_size']  # the GAN's input size is 256*256
style_dim = config['gen']['style_dim']
encode = munit.gen_a.encode
style_encode = munit.gen_b.encode
decode = munit.gen_b.decode


# process the munit's input
# warning: this is very import configuration
# transform = transforms.Compose([transforms.Resize((new_size, new_size)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([transforms.Resize(new_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# function generator use to generate transformed images by MUNIT
def generator(img, style):
    with torch.no_grad():
        img = Variable(transform(img).unsqueeze(0).cuda())
        s = Variable(style.unsqueeze(0).cuda())
        content, _ = encode(img)

        outputs = decode(content, s)
        outputs = (outputs + 1) / 2.
        del img
        del s
        del content
        return outputs.data


# process the generated image from munit
def preprocess_transformed_images(original_image):
    tensor = original_image.view(1, original_image.size(0), original_image.size(1), original_image.size(2))
    tensor = tensor.clone()

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(img):
        norm_ip(img, float(img.min()), float(img.max()))

    norm_range(tensor)
    tensor = tensor.squeeze()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    img = process_input(img)
    del ndarr
    del tensor
    return img


######################
# create some functions to calculate
######################
def calculate_uncovered_knc_sections():
    """
    Calculate the number of uncovered sections on the KNC criterion
    :return:
    """
    return np.sum([np.count_nonzero(knc_cov_dict[layer.name] == 0) for layer in layer_to_compute])


def get_new_covered_knc_sections(intermediate_layer_outputs, cov_dict):
    # cov_dict = deepcopy(knc_cov_dict)
    new_covered_sections = 0
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        output = intermediate_layer_output[0]
        layer = layer_to_compute[i]
        bins = layer_bounds_bins[layer.name]
        for id_neuron in range(output.shape[-1]):
            _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
            if _bin >= nb_part or cov_dict[layer.name][id_neuron][_bin]:
                continue
            new_covered_sections = new_covered_sections + 1
            cov_dict[layer.name][id_neuron][_bin] = True
    return new_covered_sections


def update_knc(intermediate_layer_outputs):
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        layer = layer_to_compute[i]
        bins = layer_bounds_bins[layer.name]
        output = intermediate_layer_output[0]
        for id_neuron in range(output.shape[-1]):
            _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
            if _bin >= nb_part or knc_cov_dict[layer.name][id_neuron][_bin]:
                continue
            knc_cov_dict[layer.name][id_neuron][_bin] = True


def current_knc_coverage():
    """
    Calculate the current K-Multi Section Neuron Coverage
    :return:
    """
    covered = 0
    total = 0
    for layer in layer_to_compute:
        covered = covered + np.count_nonzero(knc_cov_dict[layer.name])
        total = total + np.size(knc_cov_dict[layer.name])
    return covered / float(total)


# the fitness function
# def fitness_function(original_images, original_preds, theoretical_uncovered_sections, style):
#     transformed_images = [generator(img, style)[0] for img in original_images]
#     del original_images
#     transformed_images = [preprocess_transformed_images(img) for img in transformed_images]
#     # transformed_images = np.vstack([transformed_images])
#
#     steps = deque()
#     transformed_preds = []
#     # cov_dict = deepcopy(knc_cov_dict)
#     cov_dict = deepcopy(nbc_cov_dict)
#     new_covered_sections = 0
#     # for i in range(transformed_images.shape[0]):
#     for image in transformed_images:
#         # internal_outputs = intermediate_model.predict(transformed_images[i: i+1])
#         internal_outputs = intermediate_model.predict(image[np.newaxis, ...])
#         img = internal_outputs[-1]
#         intermediate_outputs = internal_outputs[0:-1]
#
#         del internal_outputs
#
#         if not len(steps):
#             for _ in xrange(Chauffeur_timesteps):
#                 steps.append(img)
#
#         steps.popleft()
#         steps.append(img)
#         timestepped_x = np.empty((1, Chauffeur_timesteps, img.shape[1]))
#         for i, img in enumerate(steps):
#             timestepped_x[0, i] = img
#
#         pred = Chauffeur_LSTM.predict_on_batch(timestepped_x)[0, 0] / Chauffeur_scale
#
#         transformed_preds.append(pred)
#         # new_covered_sections += get_new_covered_sections(intermediate_outputs, cov_dict)
#         new_covered_sections += get_new_covered_nbc_sections(intermediate_outputs, cov_dict)
#         del img
#         del intermediate_outputs
#
#     logger.info(new_covered_sections)
#     # logger.info(len(transformed_preds))
#     transformed_preds = np.asarray(transformed_preds)
#     o1 = float(new_covered_sections) / float(theoretical_uncovered_sections)
#     o2 = np.average(np.abs(transformed_preds - original_preds))
#     o2 = o2 / (o2 + 1)  # normalize
#     logger.info("the o1 is {}".format(o1))
#     logger.info("the o2 is {}".format(o2))
#     del new_covered_sections
#     del transformed_preds
#     del transformed_images
#     del cov_dict
#     return o1 + o2


# the fitness function
def fitness_function_1(original_images, original_preds, theoretical_uncovered_sections, style):
    steps = deque()
    transformed_preds = []
    # cov_dict = deepcopy(knc_cov_dict)
    cov_dict = deepcopy(knc_cov_dict)
    new_covered_sections = 0
    logger.info("do prediction")
    for img in original_images:
        logger.info("begin generating driving scenes")
        transformed_image = generator(img, style)[0]
        transformed_image = preprocess_transformed_images(transformed_image)[np.newaxis, ...]
        logger.info("finish generating driving scenes")

        logger.info("obtain internal outputs")
        internal_outputs = intermediate_model.predict(transformed_image)
        img = internal_outputs[-1]
        intermediate_outputs = internal_outputs[0:-1]
        logger.info("finish obtaining internal outputs")


        del internal_outputs

        if not len(steps):
            for _ in xrange(Chauffeur_timesteps):
                steps.append(img)

        steps.popleft()
        steps.append(img)
        timestepped_x = np.empty((1, Chauffeur_timesteps, img.shape[1]))
        for i, img in enumerate(steps):
            timestepped_x[0, i] = img

        pred = Chauffeur_LSTM.predict_on_batch(timestepped_x)[0, 0] / Chauffeur_scale


        transformed_preds.append(pred)
        logger.info("calculate coverage")
        new_covered_sections += get_new_covered_knc_sections(intermediate_outputs, cov_dict)
        logger.info("finish calculating coverage")

        del intermediate_outputs

    logger.info(new_covered_sections)
    # logger.info(len(transformed_preds))
    transformed_preds = np.asarray(transformed_preds)
    o1 = float(new_covered_sections) / float(theoretical_uncovered_sections)
    o2 = np.average(np.abs(transformed_preds - original_preds))
    o2 = o2 / (o2 + 1)  # normalize
    logger.info("the o1 is {}".format(o1))
    logger.info("the o2 is {}".format(o2))
    del new_covered_sections
    del transformed_preds
    del cov_dict
    return o1 + o2


# the wrapper of fitness function
def fitness_function_wrapper(original_images, original_preds, theoretical_uncovered_sections):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return fitness_function_1(original_images, original_preds, theoretical_uncovered_sections, *args, **kwargs)
        return wrapper
    return decorator


def count_error_behaviors(ori_preds, preds):
    count = 0
    for i in range(len(ori_preds)):
        if np.abs(ori_preds[i] - preds[i]) > 0.2 \
                or (ori_preds[i] < 0 and preds[i] > 0) or (ori_preds[i] > 0 and preds[i] < 0):
            count = count + 1
    return count


# def update_history(original_images, style):
#     transformed_images = [generator(img, style)[0] for img in original_images]
#     del original_images
#     transformed_images = [preprocess_transformed_images(img) for img in transformed_images]
#     transformed_images = np.vstack([transformed_images])
#
#     steps = deque()
#     transformed_preds = []
#     for i in range(transformed_images.shape[0]):
#         internal_outputs = intermediate_model.predict(transformed_images[i: i + 1])
#         img = internal_outputs[-1]
#         intermediate_outputs = internal_outputs[0:-1]
#
#         del internal_outputs
#
#         if not len(steps):
#             for _ in xrange(Chauffeur_timesteps):
#                 steps.append(img)
#
#         steps.popleft()
#         steps.append(img)
#         timestepped_x = np.empty((1, Chauffeur_timesteps, img.shape[1]))
#         for i, img in enumerate(steps):
#             timestepped_x[0, i] = img
#
#         pred = Chauffeur_LSTM.predict_on_batch(timestepped_x)[0, 0] / Chauffeur_scale
#
#         transformed_preds.append(pred)
#         # update_knc(intermediate_outputs)
#         update_nbc(intermediate_outputs)
#         del img
#         del intermediate_outputs
#     return transformed_preds

def update_history_1(original_images, style):
    steps = deque()
    transformed_preds = []
    for img in original_images:
        transformed_image = generator(img, style)[0]
        transformed_image = preprocess_transformed_images(transformed_image)[np.newaxis, ...]

        internal_outputs = intermediate_model.predict(transformed_image)
        img = internal_outputs[-1]
        intermediate_outputs = internal_outputs[0:-1]

        del internal_outputs
        if not len(steps):
            for _ in xrange(Chauffeur_timesteps):
                steps.append(img)

        steps.popleft()
        steps.append(img)
        timestepped_x = np.empty((1, Chauffeur_timesteps, img.shape[1]))
        for i, img in enumerate(steps):
            timestepped_x[0, i] = img

        pred = Chauffeur_LSTM.predict_on_batch(timestepped_x)[0, 0] / Chauffeur_scale

        transformed_preds.append(pred)
        update_knc(intermediate_outputs)
        del img
        del intermediate_outputs
    return transformed_preds


def testing():
    with open(original_preds_cache_path, 'rb') as f:
        original_preds = pickle.load(f)
    images_path = [(test_image_paths + image_file) for image_file in sorted(os.listdir(test_image_paths))
                   if image_file.endswith(".jpg")]
    orig_images_for_transform = [Image.open(path).convert('RGB') for path in images_path]
    # original_preds = original_preds[0:5]

    # nb_images = len(orig_images_for_transform)

    iteration = 0

    # print(current_knc_coverage())
    print(current_knc_coverage())

    while True:
        logger.info("the {nb_iter} begin".format(nb_iter=iteration))
        # theoretical_uncovered_sections = nb_neurons * nb_images
        nb_uncovered_sections = calculate_uncovered_knc_sections()
        theoretical_uncovered_sections = calculate_uncovered_knc_sections()

        theoretical_uncovered_sections = theoretical_uncovered_sections \
            if theoretical_uncovered_sections <= nb_uncovered_sections else nb_uncovered_sections

        print(theoretical_uncovered_sections)

        @fitness_function_wrapper(orig_images_for_transform, original_preds, theoretical_uncovered_sections)
        def fitness(style):
            pass

        search_handler = EAEngine(style_dim=style_dim, fitness_func=fitness, logger=logger)
        best = search_handler.run(150)

        transformed_preds = update_history_1(orig_images_for_transform, best)

        logger.info("the {nb_iter} finish".format(nb_iter=iteration))
        logger.info("current k-multi section coverage is {}".format(current_knc_coverage()))
        # logger.info("current neuron boundary coverage is {}".format(current_knc_coverage()))
        logger.info("the best style code is {}".format(best))
        logger.info("the number of error behaviors is {}".format(count_error_behaviors(original_preds, transformed_preds)))

        with open('/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/knc_coverage_cache_snow_night_random.pkl', 'wb') \
                as f:
            pickle.dump(knc_cov_dict, f, pickle.HIGHEST_PROTOCOL)

        iteration += 1

        if iteration == 4:
            # Terminate the algorithm when the coverage is greater than threshold
            # or the number of iterations is euqal to ten
            break


if __name__ == '__main__':
    testing()
