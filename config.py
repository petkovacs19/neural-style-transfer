import numpy as np

class Config(object):
    height = 300
    width = 400
    noise_ratio = 0.6
    channels = 3
    means = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))