# coding = utf-8
import os
import cv2
import yaml
import numpy as np
from PIL import Image


def process_input(image):
    img = cv2.resize(image, (320, 240))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img[120:240, :, :]
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img = ((img - (255.0 / 2)) / 255.0)

    return img


def create_original_data(test=0):
    # test == 1: return test images
    # test == 0: return train images
    if not test:
        if os.path.exists('/home/test/program/self-driving/testing/cache/Chauffeur/original_train_center_images.npy'):
            return np.load('/home/test/program/self-driving/testing/cache/Chauffeur/original_train_center_images.npy')
        path = '/home/test/program/self-driving/dataset/train/center/'
        images_path = [(path + image_file) for image_file in sorted(os.listdir(path)) if image_file.endswith(".jpg")][20000:]
        print len(images_path)
        images = [cv2.imread(image_path) for image_path in images_path]
        images = [process_input(img) for img in images]
        images = np.vstack([images])
        np.save('/home/test/program/self-driving/testing/cache/Chauffeur/original_train_center_images.npy', images)
        return images
    else:
        if os.path.exists('/home/test/program/self-driving/testing/cache/Chauffeur/original_test_center_images.npy'):
            return np.load('/home/test/program/self-driving/testing/cache/Chauffeur/original_test_center_images.npy')
        path = '/home/test/program/self-driving/dataset/test/center/'
        images_path = [(path + image_file) for image_file in sorted(os.listdir(path)) if image_file.endswith(".jpg")]
        print len(images_path)
        images = [cv2.imread(image_path) for image_path in images_path]
        images = [process_input(img) for img in images]
        images = np.vstack([images])
        np.save('/home/test/program/self-driving/testing/cache/Chauffeur/original_test_center_images.npy', images)
        return images


# load yaml file
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def data_generator(data, original_preds, batch_size):
    generate_state = 0
    while 1:
        if generate_state + batch_size > len(data):
            paths = data[generate_state: len(data)]
            preds = original_preds[generate_state : len(data)]
            generate_state = 0
        else:
            paths = data[generate_state: generate_state + batch_size]
            preds = original_preds[generate_state: generate_state + batch_size]
            generate_state += batch_size
        yield paths, preds


# generate data batch in a sequential manner for testing
def generate_data_batch_by_sequential(path, original_preds, batch_size):
    test_center_images = [(path + image_file) for image_file in sorted(os.listdir(path))
                          if image_file.endswith(".jpg")]
    generator = data_generator(test_center_images, original_preds, batch_size)

    return generator

