# coding=utf-8
"""
Copy from https://github.com/udacity/self-driving-car/blob/master/steering-models/evaluation/chauffeur.py
"""
from __future__ import print_function
import numpy as np
import cv2
from keras.models import model_from_json
from collections import deque
from keras import backend as K
import os
import csv
import pickle


class ChauffeurModel(object):
    def __init__(self,
                 cnn_json_path,
                 cnn_weights_path,
                 lstm_json_path,
                 lstm_weights_path):

        self.cnn = self.load_from_json(cnn_json_path, cnn_weights_path)
        self.encoder = self.load_encoder(cnn_json_path, cnn_weights_path)
        self.lstm = self.load_from_json(lstm_json_path, lstm_weights_path)

        # hardcoded from final submission model
        self.scale = 16.
        self.timesteps = 100

        self.steps = deque()

    def load_encoder(self, cnn_json_path, cnn_weights_path):
        model = self.load_from_json(cnn_json_path, cnn_weights_path)
        model.load_weights(cnn_weights_path)

        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

        return model

    def load_from_json(self, json_path, weights_path):
        model = model_from_json(open(json_path, 'r').read())
        model.load_weights(weights_path)
        return model

    def make_cnn_only_predictor(self):
        def predict_fn(img):
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:,:,0] = cv2.equalizeHist(img[:,:,0])
            img = ((img-(255.0/2))/255.0)

            return self.cnn.predict_on_batch(img.reshape((1, 120, 320, 3)))[0, 0] / self.scale

        return predict_fn

    def make_stateful_predictor(self):
        steps = deque()

        def predict_fn(img):
            # preprocess image to be YUV 320x120 and equalize Y histogram
            img = cv2.resize(img, (320, 240))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = img[120:240, :, :]
            img[:, :, 0] = cv2.equalizeHist(img[:,:,0])
            img = ((img-(255.0/2))/255.0)

            # apply feature extractor
            img = self.encoder.predict_on_batch(img.reshape((1, 120, 320, 3)))

            # print(len(steps))
            # initial fill of timesteps
            if not len(steps):
                for _ in xrange(self.timesteps):
                    steps.append(img)

            # put most recent features at end
            steps.popleft()
            steps.append(img)

            timestepped_x = np.empty((1, self.timesteps, img.shape[1]))
            for i, img in enumerate(steps):
                timestepped_x[0, i] = img

            return self.lstm.predict_on_batch(timestepped_x)[0, 0] / self.scale

        return predict_fn
    # def predict_fn(self, img):
    #     steps = self.steps
    #     # preprocess image to be YUV 320x120 and equalize Y histogram
    #     img = cv2.resize(img, (320, 240))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #     img = img[120:240, :, :]
    #     img[:, :, 0] = cv2.equalizeHist(img[:,:,0])
    #     img = ((img-(255.0/2))/255.0)
    #
    #     # apply feature extractor
    #     img = self.encoder.predict_on_batch(img.reshape((1, 120, 320, 3)))
    #
    #     # print(len(steps))
    #     # initial fill of timesteps
    #     if not len(steps):
    #         for _ in xrange(self.timesteps):
    #             steps.append(img)
    #
    #     # put most recent features at end
    #     steps.popleft()
    #     steps.append(img)
    #
    #     timestepped_x = np.empty((1, self.timesteps, img.shape[1]))
    #     for i, img in enumerate(steps):
    #         timestepped_x[0, i] = img
    #
    #     return self.lstm.predict_on_batch(timestepped_x)[0, 0] / self.scale


if __name__ == '__main__':
    cnn_json_path = './pretrained/cnn.json'
    cnn_weights_path = './pretrained/cnn.weights'
    lstm_json_path = './pretrained/lstm.json'
    lstm_weights_path = './pretrained/lstm.weights'


    def make_predictor():
        K.set_learning_phase(0)
        model = ChauffeurModel(cnn_json_path, cnn_weights_path, lstm_json_path, lstm_weights_path)
        return model.make_stateful_predictor()

    model = make_predictor()

    def calc_rmse(yhat, label):
        mse = 0.
        count = 0
        if len(yhat) != len(label):
            return -1
        for i in xrange(len(yhat)):
            count += 1
            predict_steering = yhat[i]
            steering = label[i]
            mse += (float(steering) - float(predict_steering)) ** 2
        # return (mse / count) ** 0.5
        return (mse / count)

    def count_error():
        import pickle
        import csv
        import os
        with open('/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/original_preds.pkl', 'rb') as f:
            original = pickle.load(f)

        with open('/home/test/program/self-driving/dataset/test/CH2_final_evaluation.csv', 'rb') as f:
            label2 = list(csv.reader(f, delimiter=',', quotechar='|'))
        label2 = label2[1:]

        truth = {}
        for i in label2:
            truth[i[0] + '.jpg'] = i[1]

        file_list = []
        for image_file in sorted(os.listdir('/home/test/program/self-driving/dataset/test/center/')):
            if image_file.endswith('.jpg'):
                file_list.append(image_file)

        preds = []
        labels = []
        for i, f in enumerate(file_list):
            preds.append(original[i])
            labels.append(truth[f])

        rmse = calc_rmse(preds, labels)
        # the result is 0.09165
        print("rmse:" + str(rmse))

        MSE = rmse ** 2

        yhats = []
        base_path = '/home/test/program/self-driving/Experimental_Result/Chauffeur/snowy/knc/style_0/'
        for i in range(0, 5614):
            print(i)
            img = cv2.imread(os.path.join(base_path, '{}.png'.format(i)))
            yhats.append(model(img))

        with open(
                '/home/test/program/self-driving/Experimental_Result/Chauffeur/snowy/knc/'
                'error/style_0/steering_angles.pkl', 'wb') as f:
            pickle.dump(yhats, f, pickle.HIGHEST_PROTOCOL)

        counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(labels)):
            predict_steering = yhats[i]
            steering = labels[i]
            mse = (float(steering) - float(predict_steering)) ** 2
            for i in range(1, 11):
                if mse > float(i) * MSE:
                    counts[i - 1] += 1
        print(counts)

    # count_error()

    # img1 = cv2.imread('/home/test/program/self-driving/testing/test2.png')
    # print(model(img1))
    # img2 = cv2.imread('/home/test/program/self-driving/testing/test2.png')
    # import os
    # file_list = []
    # for image_file in sorted(os.listdir('/home/test/program/self-driving/Experimental_Result/Chauffeur/snowy/nbc/style_3/')):
    #     if image_file.endswith('.jpg'):
    #         file_list.append(image_file)
    #
    # import pickle
    # with open('/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/original_preds.pkl', 'rb') as f:
    #     original = pickle.load(f)
    #
    # count = 0
    # transformed_preds = []
    # for f in file_list:
    #     seed_image = cv2.imread(os.path.join('/home/test/program/self-driving/Experimental_Result/Chauffeur/snowy/nbc/style_3/', f))
    #     # yhat = model.predict_fn(seed_image)
    #     yhat = model(seed_image)
    #     transformed_preds.append(yhat)
    #     if count % 500 == 0:
    #         print("processed images:" + str(count) + "total:" + str(5614))
    #     count = count + 1
    #
    # # count error behaviors
    # count_our = 0
    # count_10 = 0
    # count_20 = 0
    # count_30 = 0
    # count_40 = 0
    # error_id = []
    # for i in range(len(original)):
    #     ori_angle = original[i] * 25
    #     transform_angle = transformed_preds[i] * 25
    #     if np.abs(original[i] - transformed_preds[i]) > 0.2 \
    #             or (original[i] < 0 and transformed_preds[i] > 0) or (original[i] > 0 and transformed_preds[i] < 0):
    #         count_our = count_our + 1
    #         error_id.append(i)
    #     if np.abs(ori_angle - transform_angle) > 10:
    #         count_10 = count_10 + 1
    #     if np.abs(ori_angle - transform_angle) > 20:
    #         count_20 = count_20 + 1
    #     if np.abs(ori_angle - transform_angle) > 30:
    #         count_30 = count_30 + 1
    #     if np.abs(ori_angle - transform_angle) > 40:
    #         count_40 = count_40 + 1
    #
    # print("the number of error behaviors is {} when threshold is 0.2".format(count_our))
    # print("the number of error behaviors is {} when threshold is 10".format(count_10))
    # print("the number of error behaviors is {} when threshold is 20".format(count_20))
    # print("the number of error behaviors is {} when threshold is 30".format(count_30))
    # print("the number of error behaviors is {} when threshold is 40".format(count_40))
    #
    # folder_path = '/home/test/program/self-driving/Experimental_Result/Chauffeur/snowy/nbc/error/style_3/'
    # error_id_path = os.path.join(folder_path, 'error_id.pkl')
    # with open(error_id_path, 'wb') as f:
    #     pickle.dump(error_id, f, pickle.HIGHEST_PROTOCOL)
    # K.set_learning_phase(0)
    # model = ChauffeurModel(cnn_json_path, cnn_weights_path, lstm_json_path, lstm_weights_path)

    # original images calculate
    # def calc_rmse(yhat, label):
    #     mse = 0.
    #     count = 0
    #     if len(yhat) != len(label):
    #         return -1
    #     for i in xrange(len(yhat)):
    #         count += 1
    #         predict_steering = yhat[i]
    #         steering = label[i]
    #         mse += (float(steering) - float(predict_steering)) ** 2
    #     return (mse / count) ** 0.5
    #
    # import os
    # import csv
    #
    # seed_inputs1 = '/home/test/program/self-driving/dataset/hmb3_jpg/'
    # seed_labels1 = '/home/test/program/self-driving/dataset/hmb3_jpg/hmb3_steering.csv'
    seed_inputs2 = '/home/test/program/self-driving/dataset/test/center/'
    seed_labels2 = '/home/test/program/self-driving/dataset/test/CH2_final_evaluation.csv'
    #
    # filelist1 = []
    # for image_file in sorted(os.listdir(seed_inputs1)):
    #     if image_file.endswith(".jpg"):
    #         filelist1.append(image_file)
    truth = {}
    # with open(seed_labels1, 'rb') as csvfile1:
    #     label1 = list(csv.reader(csvfile1, delimiter=',', quotechar='|'))
    # label1 = label1[1:]
    # for i in label1:
    #     truth[i[0] + ".jpg"] = i[1]
    #
    filelist2 = []
    for image_file in sorted(os.listdir(seed_inputs2)):
        if image_file.endswith(".jpg"):
            filelist2.append(image_file)
    with open(seed_labels2, 'rb') as csvfile2:
        label2 = list(csv.reader(csvfile2, delimiter=',', quotechar='|'))
    label2 = label2[1:]
    #
    for i in label2:
        truth[i[0] + ".jpg"] = i[1]
    #
    # yhats = []
    labels = []
    # count = 0
    # total = len(filelist1) + len(filelist2)
    # for f in filelist1:
    #     seed_image = cv2.imread(os.path.join(seed_inputs1, f))
    #     yhat = model(seed_image)
    #     yhats.append(yhat)
    #     labels.append(truth[f])
    #     if count % 500 == 0:
    #         print("processed images: " + str(count) + " total: " + str(total))
    #     count = count + 1
    #
    for f in filelist2:
    #     seed_image = cv2.imread(os.path.join(seed_inputs2, f))
    #     yhat = model(seed_image)
    #     yhats.append(yhat)
        labels.append(truth[f])
    #     if count % 500 == 0:
    #         print("processed images: " + str(count) + " total: " + str(total))
    #     count = count + 1
    with open('/home/test/program/self-driving/testing/cache/Chauffeur/test_outputs/steering_angles.pkl', 'rb') as f:
        original_preds = pickle.load(f)
        mse = calc_rmse(original_preds, labels)
        print(mse)
    with open('/home/test/program/self-driving/testing/cache/Dave_orig/test_outputs/steering_angles.pkl', 'rb') as f:
        original_preds = pickle.load(f)
        mse = calc_rmse(original_preds, labels)
        print(mse)
    with open('/home/test/program/self-driving/testing/cache/Dave_dropout/test_outputs/steering_angles.pkl', 'rb') as f:
        original_preds = pickle.load(f)
        mse = calc_rmse(original_preds, labels)
        print(mse)

    # # TEST: RMSE:0.091
    # # HMB_3: RMSE:0.091281230445
    # # ALL: RMSE:0.0920
    # print("mse: " + str(mse))

    # import os
    # import pickle
    # base_path = '/home/test/program/self-driving/Experimental_Result/Chauffeur/rainy/knc/style_3/'
    #
    # yhats = []
    # for i in range(0, 5614):
    #     print(i)
    #     img = cv2.imread(os.path.join(base_path, '{}.png'.format(i)))
    #     yhat = model(img)
    #     yhats.append(yhat)
    #
    # with open('/home/test/program/self-driving/Experimental_Result/Chauffeur/rainy/knc/error/style_3/steering_angles.pkl', 'wb') as f:
    #     pickle.dump(yhats, f, pickle.HIGHEST_PROTOCOL)

