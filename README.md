# TACTIC

This repository contains code for the paper "**Testing DNN-based Autonomous Driving Systems under Critical Environmental Conditions**"
## Dependencies
- OS: Ubuntu 16.04
- Packages: Tensorflow 1.2.0; Keras 1.2.2; and Pytorch 0.4.1.

Note: TACTIC relies on MUNIT to produce testing driving scenes. See more details about MUNIT in the repository  [https://github.com/NVlabs/MUNIT](https://github.com/NVlabs/MUNIT).

## Overview of the Repository

The major contents of this repository are:

- [example_scenes/](https://github.com/SEG-DENSE/TACTIC/tree/main/example_scenes) contains the examples of generated driving scenes by our method
- [models/](https://github.com/SEG-DENSE/TACTIC/tree/main/models) contains the implementation of subject DNN-based autonomous driving systems
- [munit/](https://github.com/SEG-DENSE/TACTIC/tree/main/munit) contains code for MUNIT model
- [testing/](https://github.com/SEG-DENSE/TACTIC/tree/main/testing) contains code for our experiments

## Models

### DNN-based ADSs

In our experiments, we considered three popular DNN-based ADs, which have been widely used in previous work, namely Dave-orig, Dave-dropout, and Chauffuer.

For Dave-orig and Dave-dropout, the implementations of the models and the corresponding saved weights can be found in https://github.com/peikexin9/deepxplore/tree/master/Driving

For Chauffeur, the implementation of the model and the corresponding saved weight can be found in https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/chauffeur.

### MUNIT Models

TACTIC relies on MUNIT to encode the environmental condition space and generate testing driving scenes. The details of MUNIT can be found in [https://github.com/NVlabs/MUNIT](https://github.com/NVlabs/MUNIT). All save weights of MUNIT models used in our experiments can be downloaded from [here](https://1drv.ms/u/s!ArfDZDT3m0qHg0LV1LXttx_YS3k8?e=P6Q4kr)

## Datasets

- [Udacity Dataset](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2.): Dataset in the Udacity self-driving car challenge
