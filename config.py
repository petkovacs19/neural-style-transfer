import numpy as np

class Config(object):
    height = 1134
    width = 1512
    noise_ratio = 0.6
    channels = 3
    means = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
