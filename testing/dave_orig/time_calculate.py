from models.dave.networks import Dave_orig
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from munit.munit import MUNIT
from munit.utils import get_config
from testing.engine import RandomSearch, EAEngine
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from copy import deepcopy
import torch
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
handler = logging.FileHandler("./logger/knc_snow_night_ES_time_cost.txt")
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

with open('/home/test/program/self-driving/testing/cache/Dave_orig/train_outputs/layer_bounds_bin.pkl', 'rb') as f:
    layer_bounds_bins = pickle.load(f)
with open('/home/test/program/self-driving/testing/cache/Dave_orig/test_outputs/knc_coverage.pkl', 'rb') as f:
    knc_cov_dict = pickle.load(f)
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


if __name__ == '__main__':
    images_path = [(test_image_paths + image_file) for image_file in sorted(os.listdir(test_image_paths))
                   if image_file.endswith(".jpg")]
    orig_images_for_transform = [Image.open(path).convert('RGB') for path in images_path]
    print "begin transformation"
    style = Variable(torch.randn(8, 1, 1).unsqueeze(0).cuda())
    transformed_image = generator(orig_images_for_transform, style)[0]
    logger.info("finish generating driving scenes")

    # logger.info("obtain internal outputs")
    # internal_outputs = intermediate_model.predict(transformed_image)
    # intermediate_outputs = internal_outputs[0:-1]
    # preds.append(internal_outputs[-1][0][0])
    # logger.info("finish obtaining internal outputs")
    #
    # logger.info("calculate coverage")
    # new_covered_sections += get_new_covered_knc_sections(intermediate_outputs, cov_dict)
    # logger.info("finish calculating coverage")