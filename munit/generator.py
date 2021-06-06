# coding=utf-8
"""
Generate images based MUNIT
"""
import utils
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from munit import MUNIT
import torchvision.utils as vutils


################
# Parameters
################
# config_path = '/home/test/program/self-driving/munit/configs/rainy.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/rainy/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/night.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/night/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/snowy.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/snowy/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/sunny.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/sunny/gen_01250000.pt'
# checkpoint_path = '/home/test/program/MUNIT-master/outputs/day2snow/checkpoints/gen_00260000.pt'
config_path = '/home/test/program/self-driving/munit/configs/snow_night.yaml'
checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/snow_night/gen_01000000.pt'

config = utils.get_config(config_path)

model = MUNIT(config)
try:
    state_dict = torch.load(checkpoint_path)
    model.gen_a.load_state_dict(state_dict['a'])
    model.gen_b.load_state_dict(state_dict['b'])
except:
    raise RuntimeError('load model failed')

model.cuda()
new_size = config['new_size']
style_dim = config['gen']['style_dim']
encode = model.gen_a.encode
style_encode = model.gen_b.encode
decode = model.gen_b.decode

#
# import torch
# import math
# irange = range

# generate transformed images
def generator(image, style):
    with torch.no_grad():
        transform = transforms.Compose([transforms.Resize((new_size, new_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        image = Variable(transform(image).unsqueeze(0).cuda())
        s = Variable(style.unsqueeze(0).cuda())

        #print image.shape

        content, _ = encode(image)

        outputs = decode(content, s)
        outputs = (outputs + 1) / 2.
        #vutils.save_image(outputs.data, './tt.jpg', padding=0, normalize=False)
        return outputs.data


if __name__ == '__main__':
    # img = Image.open('../dataset/test/center/1479425441182877835.jpg')
    # new_img = img.resize((256, 256), Image.BILINEAR)
    # for i in range(30):
    #     print i
    import os
    import numpy as np
    import cv2
    test_image_paths = '/home/test/program/self-driving/dataset/test/center/'
    image_save_path = '/home/test/program/self-driving-experiments/Experimental_Result/Image_example/snow_night/'
    images_path = [(test_image_paths + image_file) for image_file in sorted(os.listdir(test_image_paths))
                   if image_file.endswith(".jpg")]
    # index = np.random.randint(0, 5614, 10)
    # images_path = [images_path[i] for i in index]
    # orig_images_for_transform = [Image.open(path).convert('RGB') for path in images_path]
    with torch.no_grad():
        transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for i in range(1, 23):
            orig_images_for_transform = Image.open(images_path[i]).convert('RGB')
            image = Variable(transform(orig_images_for_transform).unsqueeze(0).cuda())
            image_save_path_1 = os.path.join(image_save_path, 'rainy_{}'.format(i))
            os.mkdir(image_save_path_1)
            orig_path = os.path.join(image_save_path_1, 'orig.png')
            orig_images_for_transform.save(orig_path)
            for j in range(50):
                style = Variable(torch.randn(8, 1, 1).unsqueeze(0).cuda())
                # print style

                content, _ = encode(image)

                outputs = decode(content, style)
                outputs = (outputs + 1) / 2.
                file_path = os.path.join(image_save_path_1, '{}.png'.format(j))
                vutils.save_image(outputs.data, file_path, padding=0, normalize=True)
    #     for i in range(10):
    #         image_save_path_1 = os.path.join(image_save_path, 'rainy_{}'.format(i + 11))
    #         os.mkdir(image_save_path_1)
    #         orig = orig_images_for_transform[i]
    #         orig_path = os.path.join(image_save_path_1, 'orig.png')
    #         orig_images_for_transform[i].save(orig_path)
    #         image = Variable(transform(orig_images_for_transform[i]).unsqueeze(0).cuda())
    #         for j in range(10):
    #             style = Variable(torch.randn(8, 1, 1).unsqueeze(0).cuda())
    #             # print style
    #
    #             content, _ = encode(image)
    #
    #             outputs = decode(content, style)
    #             outputs = (outputs + 1) / 2.
    #             file_path = os.path.join(image_save_path_1, '{}.png'.format(j))
    #             vutils.save_image(outputs.data, file_path, padding=0, normalize=True)

        # image = Variable(
        #     transform(Image.open('../dataset/test/center/1479425441182877835.jpg').convert('RGB')).unsqueeze(0).cuda())
        # # style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None
        # # Start testing
        # content, _ = encode(image)
        # style = Variable(torch.randn(10, style_dim, 1, 1).cuda())
        # for j in range(4):
        #     s = style[j].unsqueeze(0)
        #     outputs = decode(content, s)
        #     outputs = (outputs + 1) / 2.
        #     # print outputs.data.size()
        #     path = './night{}.png'.format(j)
        #     vutils.save_image(outputs.data, path, padding=0, normalize=True)
