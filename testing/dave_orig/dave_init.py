from models.dave.networks import Dave_orig
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import pickle
import os
weights_path = '/home/test/program/self-driving/models/dave/pretrained/dave_orig.h5'

model = Dave_orig(load_weights=True, weights_path=weights_path)
model.summary()

layer_to_compute = [layer for layer in model.layers
                    if all(ex not in layer.name for ex in ['flatten', 'input'])][0:-2]

outputs_layer = [layer.output for layer in layer_to_compute]
outputs_layer.append(model.layers[-1].output)
intermediate_model = Model(input=model.input, output=outputs_layer)


def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data


# layer_bounds = {}
# layer_bounds_bin = {}
# for layer in layer_to_compute:
#     layer_bounds[layer.name] = None
with open('/home/test/program/self-driving/testing/cache/Dave_orig/train_outputs/layer_bounds.pkl', 'rb') as f:
    layer_bounds = pickle.load(f)

with open('/home/test/program/self-driving/testing/cache/Dave_orig/train_outputs/layer_bounds_bin.pkl', 'rb') as f:
    layer_bounds_bin = pickle.load(f)


def update_bounds(intermediate_layer_outputs):
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        output = intermediate_layer_output[0]
        layer = layer_to_compute[i]
        if layer_bounds[layer.name] is None:
            low_bound = [np.inf] * output.shape[-1]
            high_bound = [-np.inf] * output.shape[-1]
        else:
            (low_bound, high_bound) = layer_bounds[layer.name]
        for id_neuron in range(output.shape[-1]):
            val = np.mean(output[..., id_neuron])
            if val < low_bound[id_neuron]:
                low_bound[id_neuron] = val
            if val > high_bound[id_neuron]:
                high_bound[id_neuron] = val
        layer_bounds[layer.name] = (low_bound, high_bound)


def init_bounds(img_paths):
    for i, img_path in enumerate(img_paths):
        print i
        img = preprocess_image(img_path)
        internal_outputs = intermediate_model.predict(img)
        intermediate_outputs = internal_outputs[0:-1]
        update_bounds(intermediate_outputs)

    with open('/home/test/program/self-driving/testing/cache/Dave_orig/train_outputs/layer_bounds.pkl', 'wb') as f:
        pickle.dump(layer_bounds, f, pickle.HIGHEST_PROTOCOL)

    for layer in layer_to_compute:
        (low_bound, high_bound) = layer_bounds[layer.name]
        layer_bounds_bin[layer.name] = [np.linspace(low_bound[i], high_bound[i], 1000 + 1)
                                                 for i in range(len(high_bound))]
    with open('/home/test/program/self-driving/testing/cache/Dave_orig/train_outputs/layer_bounds_bin.pkl', 'wb') as f:
        pickle.dump(layer_bounds_bin, f, pickle.HIGHEST_PROTOCOL)


knc_cov_dict = {}
nbc_cov_dict = {}
for layer in layer_to_compute:
    knc_cov_dict[layer.name] = np.zeros((layer.output.get_shape()[-1], 1000), dtype='bool')
    nbc_cov_dict[layer.name] = np.zeros((layer.output.get_shape()[-1], 2), dtype='bool')


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


def update_knc(intermediate_layer_outputs):
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        layer = layer_to_compute[i]
        bins = layer_bounds_bin[layer.name]
        output = intermediate_layer_output[0]
        for id_neuron in range(output.shape[-1]):
            _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
            if _bin >= 1000 or knc_cov_dict[layer.name][id_neuron][_bin]:
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


def init_cov(img_paths):
    preds = []
    for i, img_path in enumerate(img_paths):
        print i
        img = preprocess_image(img_path)
        internal_outputs = intermediate_model.predict(img)
        intermediate_outputs = internal_outputs[0:-1]
        preds.append(internal_outputs[-1][0][0])
        update_knc(intermediate_outputs)
        update_nbc(intermediate_outputs)

    print current_knc_coverage()
    print current_nbc_coverage()

    with open('/home/test/program/self-driving/testing/cache/Dave_orig/test_outputs/knc_coverage.pkl', 'wb') as f:
        pickle.dump(knc_cov_dict, f, pickle.HIGHEST_PROTOCOL)

    with open('/home/test/program/self-driving/testing/cache/Dave_orig/test_outputs/nbc_coverage.pkl', 'wb') as f:
        pickle.dump(nbc_cov_dict, f, pickle.HIGHEST_PROTOCOL)

    with open('/home/test/program/self-driving/testing/cache/Dave_orig/test_outputs/steering_angles.pkl', 'wb') as f:
        pickle.dump(preds, f, pickle.HIGHEST_PROTOCOL)

# # train_image_paths = '/home/test/program/self-driving/dataset/train/center/'
# test_image_paths = '/home/test/program/self-driving/dataset/test/center/'
# filelist = []
# for image_file in sorted(os.listdir(test_image_paths)):
#     if image_file.endswith(".jpg"):
#         filelist.append(os.path.join(test_image_paths, image_file))
# print len(filelist)
# init_cov(filelist)
