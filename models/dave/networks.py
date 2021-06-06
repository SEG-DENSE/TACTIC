# coding=utf-8
"""
three DNNs  based on the DAVE-2 self-driving car architecture from Nvidia with slightly different configurations
copyright: https://github.com/peikexin9/deepxplore/blob/master/Driving
"""
from keras.layers import Convolution2D, Input, Dense, Flatten, Lambda, MaxPooling2D, Dropout, Conv2D
from keras.models import Model
from keras import backend as K
import tensorflow as tf

# K.set_learning_phase(0)


###################
# config
###################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


###################
# some utils
###################
def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def normal_init(shape):
    # return K.truncated_normal(shape, stddev=0.1)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return K.variable(initial)


###################
# the DNN models
###################
# the original architecture
def Dave_orig(input_tensor=None, load_weights=False, weights_path=None):  # original dave
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2), name='block1_conv1')(input_tensor)
    x = Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2), name='block1_conv2')(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2), name='block1_conv3')(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1), name='block1_conv4')(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1), name='block1_conv5')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1164, activation='relu', name='fc1')(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(50, activation='relu', name='fc3')(x)
    x = Dense(10, activation='relu', name='fc4')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights(weights_path)

    # compiling
    m.compile(loss='mse', optimizer='adadelta')
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m


# removes the first batch normalization layer and normalizes the randomly initialized network weights
def Dave_norminit(input_tensor=None, load_weights=False, weights_path=None):  # original dave with normal initialization
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2),
                      name='block1_conv1')(input_tensor)
    x = Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2),
                      name='block1_conv2')(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2),
                      name='block1_conv3')(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1),
                      name='block1_conv4')(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1),
                      name='block1_conv5')(x)
    x = Flatten(name='flatten')(x)
    # x = Dense(1164, kernel_initializer=normal_init, activation='relu', name='fc1')(x)
    # x = Dense(100, kernel_initializer=normal_init, activation='relu', name='fc2')(x)
    # x = Dense(50, kernel_initializer=normal_init, activation='relu', name='fc3')(x)
    # x = Dense(10, kernel_initializer=normal_init, activation='relu', name='fc4')(x)
    # x = Dense(1164, init=normal_init, activation='relu', name='fc1')(x)
    # x = Dense(100, init=normal_init, activation='relu', name='fc2')(x)
    # x = Dense(50, init=normal_init, activation='relu', name='fc3')(x)
    # x = Dense(10, init=normal_init, activation='relu', name='fc4')(x)
    # x = Dense(1, name='before_prediction')
    x = Dense(1164, activation='relu', name='fc1')(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dense(50, activation='relu', name='fc3')(x)
    x = Dense(10, activation='relu', name='fc4')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name='prediction')(x)

    m = Model(input_tensor, x)
    if load_weights:
        # m.load_weights('./Model2.h5')
        m.load_weights(weights_path)

    # compiling
    m.compile(loss='mse', optimizer='adadelta')
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m


# cutting down the numbers of convo- lutional layers and fully connected layers
def Dave_dropout(input_tensor=None, load_weights=False, weights_path=None):  # simplified dave
    if input_tensor is None:
        input_tensor = Input(shape=(100, 100, 3))
    x = Convolution2D(16, 3, 3, border_mode='valid', activation='relu', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)
    x = Convolution2D(32, 3, 3, border_mode='valid', activation='relu', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool2')(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', activation='relu', name='block1_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool3')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(500, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu', name='fc2')(x)
    x = Dropout(0.25)(x)
    x = Dense(20, activation='relu', name='fc3')(x)
    x = Dense(1, name='before_prediction')(x)
    x = Lambda(atan_layer, output_shape=atan_layer_shape, name="prediction")(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights(weights_path)

    # compiling
    m.compile(loss='mse', optimizer='adadelta')
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m


def calc_rmse(yhat, label):
    mse = 0.
    count = 0
    if len(yhat) != len(label):
        print ("yhat and label have different lengths")
        return -1
    for i in xrange(len(yhat)):
        count += 1
        predicted_steering = yhat[i]
        steering = label[i]
        #print(predicted_steering)
        #print(steering)
        mse += (float(steering) - float(predicted_steering))**2.
    return mse/count


if __name__ == '__main__':
    K.set_learning_phase(0)
    model1 = Dave_orig(load_weights=True, weights_path='./pretrained/dave_orig.h5')
    model1.summary()

    model2 = Dave_norminit(load_weights=True, weights_path='./pretrained/dave_norminit.h5')
    model2.summary()

    model3 = Dave_dropout(load_weights=True, weights_path='./pretrained/Model3.h5')
    model3.summary()

    from keras.preprocessing import image
    import numpy as np
    from keras.applications.imagenet_utils import preprocess_input


    def preprocess_image(img_path, target_size=(100, 100)):
        img = image.load_img(img_path, target_size=target_size)
        input_img_data = image.img_to_array(img)
        input_img_data = np.expand_dims(input_img_data, axis=0)
        input_img_data = preprocess_input(input_img_data)
        return input_img_data


    seed_inputs2 = '/home/test/program/self-driving/dataset/test/center/'
    seed_labels2 = '/home/test/program/self-driving/dataset/test/CH2_final_evaluation.csv'

    # seed_inputs = '/home/test/program/self-driving/Experimental_Result/Dave_orig/rainy/nbc/style_0/'

    import os
    import csv

    truth = {}
    filelist1 = []
    for image_file in sorted(os.listdir(seed_inputs2)):
        if image_file.endswith(".jpg"):
            filelist1.append(image_file)

    # filelist2 = []
    # for image_file in sorted(os.listdir(seed_inputs)):
    #     if image_file.endswith(".png"):
    #         filelist2.append(image_file)

    with open(seed_labels2, 'rb') as csvfile2:
        label2 = list(csv.reader(csvfile2, delimiter=',', quotechar='|'))
    label2 = label2[1:]

    for i in label2:
        truth[i[0] + ".jpg"] = i[1]

    preds1 = []
    preds2 = []
    preds3 = []
    labels = []
    count = 0
    total = len(filelist1)
    for f in filelist1:
        yhat1 = model1.predict(preprocess_image(os.path.join(seed_inputs2, f)))[0][0]
        preds1.append(yhat1)
        # yhat2 = model2.predict(preprocess_image(os.path.join(seed_inputs2, f)))[0][0]
        # preds2.append(yhat2)
        # yhat3 = model3.predict(preprocess_image(os.path.join(seed_inputs2, f)))[0][0]
        # preds3.append(yhat3)
        labels.append(truth[f])
        if count % 500 == 0:
            print ("processed images: " + str(count) + " total: " + str(total))
        count = count + 1
    print preds1[0:10]
    print calc_rmse(preds1, labels)
    import pickle
    with open('/home/test/program/self-driving/testing/cache/Dave_orig/test_outputs/steering_angles.pkl', 'rb') as f:
        original_preds = pickle.load(f)
        print original_preds[0:10]
        print calc_rmse(original_preds, labels)
    # print preds2
    # print calc_rmse(preds2, labels)
    # print preds3
    # print calc_rmse(preds3, labels)