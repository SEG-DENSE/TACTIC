from models.dave.networks import Dave_orig
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from munit.munit import MUNIT
from munit.utils import get_config
from keras.preprocessing import image
from testing.engine import RandomSearch
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
import pickle
import logging
import os
####################
# parameters
####################
# the image path
train_image_paths = '/home/test/program/self-driving/dataset/train/center/'
test_image_paths = '/home/test/program/self-driving/dataset/test/center/'
# the cache of original images steering angles
original_preds_cache_path = '/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/original_preds.pkl'
# munit model path
# config_path = '/home/test/program/self-driving/munit/configs/snowy.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/snowy/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/night.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/night/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/rainy.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/rainy/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/sunny.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/sunny/gen_01250000.pt'
config_path = '/home/test/program/self-driving/munit/configs/snow_night.yaml'
checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/snow_night/gen_01000000.pt'
# the self-driving system's weight file
weights_path = '/home/test/program/self-driving/models/dave/pretrained/dave_orig.h5'
target_size = (100, 100)
nb_part = 1000

###################
# set logger
###################
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("./logger/nbc_snow_night_random.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

model = Dave_orig(load_weights=True, weights_path=weights_path)
model.summary()

layer_to_compute = [layer for layer in model.layers
                    if all(ex not in layer.name for ex in ['flatten', 'input'])][0:-2]

outputs_layer = [layer.output for layer in layer_to_compute]
outputs_layer.append(model.layers[-1].output)
intermediate_model = Model(input=model.input, output=outputs_layer)

with open('/home/test/program/self-driving/testing/cache/Dave_orig/train_outputs/layer_bounds.pkl', 'rb') as f:
    layer_bounds = pickle.load(f)
with open('/home/test/program/self-driving/testing/cache/Dave_orig/test_outputs/nbc_coverage.pkl', 'rb') as f:
    nbc_cov_dict = pickle.load(f)
with open('/home/test/program/self-driving/testing/cache/Dave_orig/test_outputs/steering_angles.pkl', 'rb') as f:
    original_steering_angles = pickle.load(f)

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
    img = Image.fromarray(ndarr)
    img = img.resize((target_size[1], target_size[0]))
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data

######################
# create some functions to calculate
######################
def calculate_uncovered_nbc_sections():
    """
    Calculate the number of uncovered sections on the KNC criterion
    :return:
    """
    return np.sum([np.count_nonzero(nbc_cov_dict[layer.name] == 0) for layer in layer_to_compute])


def get_new_covered_nbc_sections(intermediate_layer_outputs, cov_dict):
    # cov_dict = deepcopy(knc_cov_dict)
    new_covered_sections = 0
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        output = intermediate_layer_output[0]
        layer = layer_to_compute[i]
        (low_bound, high_bound) = layer_bounds[layer.name]
        for id_neuron in range(output.shape[-1]):
            val = np.mean(output[..., id_neuron])
            if val < low_bound[id_neuron] and not cov_dict[layer.name][id_neuron][0]:
                new_covered_sections = new_covered_sections + 1
                cov_dict[layer.name][id_neuron][0] = True
            elif val > high_bound[id_neuron] and not cov_dict[layer.name][id_neuron][1]:
                new_covered_sections = new_covered_sections + 1
                cov_dict[layer.name][id_neuron][1] = True
            else:
                continue
    return new_covered_sections


def update_nbc(intermediate_layer_outputs):
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        output = intermediate_layer_output[0]
        layer = layer_to_compute[i]
        (low_bound, high_bound) = layer_bounds[layer.name]
        for id_neuron in range(output.shape[-1]):
            val = np.mean(output[..., id_neuron])
            if val < low_bound[id_neuron] and not nbc_cov_dict[layer.name][id_neuron][0]:
                nbc_cov_dict[layer.name][id_neuron][0] = True
            elif val > high_bound[id_neuron] and not nbc_cov_dict[layer.name][id_neuron][1]:
                nbc_cov_dict[layer.name][id_neuron][1] = True
            else:
                continue


def current_nbc_coverage():
    """
    Calculate the current Neuron Boundary Coverage
    :return:
    """
    covered = 0
    total = 0
    for layer in layer_to_compute:
        covered = covered + np.count_nonzero(nbc_cov_dict[layer.name])
        total = total + np.size(nbc_cov_dict[layer.name])
    return covered / float(total)


# the fitness function
def fitness_function(original_images, original_preds, theoretical_uncovered_sections, style):
    preds = []
    cov_dict = deepcopy(nbc_cov_dict)
    new_covered_sections = 0
    for img in original_images:
        transformed_image = generator(img, style)[0]
        transformed_image = preprocess_transformed_images(transformed_image)

        internal_outputs = intermediate_model.predict(transformed_image)
        intermediate_outputs = internal_outputs[0:-1]
        preds.append(internal_outputs[-1][0][0])

        new_covered_sections += get_new_covered_nbc_sections(intermediate_outputs, cov_dict)

    logger.info(new_covered_sections)
    # logger.info(len(transformed_preds))
    transformed_preds = np.asarray(preds)
    o1 = float(new_covered_sections) / float(theoretical_uncovered_sections)
    o2 = np.average(np.abs(transformed_preds - original_preds))
    o2 = o2 / (o2 + 1)  # normalize
    logger.info("the o1 is {}".format(o1))
    logger.info("the o2 is {}".format(o2))
    del new_covered_sections
    del transformed_preds
    del cov_dict
    return 10 * o1 + o2


# the wrapper of fitness function
def fitness_function_wrapper(original_images, original_preds, theoretical_uncovered_sections):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return fitness_function(original_images, original_preds, theoretical_uncovered_sections, *args, **kwargs)
        return wrapper
    return decorator


def count_error_behaviors(ori_preds, preds):
    count = 0
    for i in range(len(ori_preds)):
        if np.abs(ori_preds[i] - preds[i]) > 0.2 \
                or (ori_preds[i] < 0 and preds[i] > 0) or (ori_preds[i] > 0 and preds[i] < 0):
            count = count + 1
    return count


def update_history(original_images, style):
    preds = []
    for img in original_images:
        transformed_image = generator(img, style)[0]
        transformed_image = preprocess_transformed_images(transformed_image)

        internal_outputs = intermediate_model.predict(transformed_image)
        intermediate_outputs = internal_outputs[0:-1]
        preds.append(internal_outputs[-1][0][0])

        update_nbc(intermediate_outputs)
    return preds


def testing():
    images_path = [(test_image_paths + image_file) for image_file in sorted(os.listdir(test_image_paths))
                   if image_file.endswith(".jpg")]
    orig_images_for_transform = [Image.open(path).convert('RGB') for path in images_path]

    iteration = 0

    print(current_nbc_coverage())

    while True:
        logger.info("the {nb_iter} begin".format(nb_iter=iteration))
        # theoretical_uncovered_sections = nb_neurons * nb_images
        theoretical_uncovered_sections = calculate_uncovered_nbc_sections()

        print(theoretical_uncovered_sections)

        @fitness_function_wrapper(orig_images_for_transform, original_steering_angles, theoretical_uncovered_sections)
        def fitness(style):
            pass

        search_handler = RandomSearch(style_dim=style_dim, fitness_func=fitness, logger=logger)
        best = search_handler.run(150)

        transformed_preds = update_history(orig_images_for_transform, best)

        logger.info("the {nb_iter} finish".format(nb_iter=iteration))
        # logger.info("current k-multi section coverage is {}".format(current_nbc_coverage()))
        logger.info("current neuron boundary coverage is {}".format(current_nbc_coverage()))
        logger.info("the best style code is {}".format(best))
        logger.info("the number of error behaviors is {}".format(count_error_behaviors(original_steering_angles, transformed_preds)))

        with open('/home/test/program/self-driving/testing/cache/Dave_orig/test_outputs/nbc_coverage_cache_snow_night_random.pkl', 'wb') \
                as f:
            pickle.dump(nbc_cov_dict, f, pickle.HIGHEST_PROTOCOL)

        iteration += 1

        if iteration == 4 or current_nbc_coverage() >= 0.40:
            # Terminate the algorithm when the coverage is greater than threshold
            # or the number of iterations is euqal to ten
            break


if __name__ == '__main__':
    testing()
