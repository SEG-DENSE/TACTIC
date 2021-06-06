import utils
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from unit import UNIT_Trainer
import torchvision.utils as vutils
import os


################
# Parameters
################
config_path = '../munit/unit_configs/snow_night.yaml'
checkpoint_path = '../munit/unit_checkpoints/snow_night/gen_01000000.pt'
# checkpoint_path = '/home/test/program/MUNIT-master/outputs/day2snow/checkpoints/gen_00260000.pt'

config = utils.get_config(config_path)

model = UNIT_Trainer(config)
try:
    state_dict = torch.load(checkpoint_path)
    model.gen_a.load_state_dict(state_dict['a'])
    model.gen_b.load_state_dict(state_dict['b'])
except:
    raise RuntimeError('load model failed')

model.cuda()
new_size = config['new_size']
encode = model.gen_a.encode
decode = model.gen_b.decode

if __name__ == '__main__':
    test_image_paths = '/home/test/program/self-driving/dataset/test/center/'
    image_save_path = '/home/test/program/self-driving-experiments/Experimental_Result/UNIT/snow_night/'
    images_path = [(test_image_paths + image_file) for image_file in sorted(os.listdir(test_image_paths))
                   if image_file.endswith(".jpg")]
    orig_images_for_transform = [Image.open(path).convert('RGB') for path in images_path]
    with torch.no_grad():
        transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    for j, image in enumerate(orig_images_for_transform):
        img = Variable(transform(image).unsqueeze(0).cuda())

        content, _ = encode(img)

        outputs = decode(content)
        outputs = (outputs + 1) / 2.
        path = os.path.join(image_save_path, '{}.png'.format(j))
        vutils.save_image(outputs.data, path, padding=0, normalize=True)