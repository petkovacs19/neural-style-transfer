import numpy as np
import tensorflow as tf
import scipy.io
from config import *
import os.path

config = Config()

def get_conv_layer(raw_layer, prev_layer):
    """
    Returns a tf.nn.conv2d layer using the weight and biases from loaded raw_layer
    """
    weight = tf.constant(raw_layer[0][0])
    raw_bias = raw_layer[0][1]
    bias = tf.constant(np.reshape(raw_bias, (raw_bias.size)))
    return tf.nn.conv2d(prev_layer,filter = weight, strides = [1,1,1,1], padding = 'SAME') + bias 


def get_vgg_graph(path = "imagenet-vgg-verydeep-19.mat"):
    """
    Returns a dictionary of tensors (layer_name) -> tensor
    NOTE! It does not contain the fully connected layers
    """
    assert os.path.exists(path), "The model does not exist! Download imagenet-vgg-verydeep-19 and specify path"
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers'][0]
    
    #create a dictionary to contain the tensorflow graph
    graph = {}
    #create the input tensor
    graph['input'] = tf.Variable(np.zeros((1, config.height, config.width, config.channels)), dtype = 'float32')
    
    #create a map  layer_name -> raw_layer_data
    layer_dict = {}
    for layer in vgg_layers:
        layer_name = layer[0][0][0][0]
        layer_dict[layer_name] = layer[0][0][2]
    
    #layer names
    layer_names = list(layer_dict.keys())
    
    #create a dictionary of layer_name -> tensorflow tensors
    for index, layer_name in enumerate(layer_names):
        prev_layer = graph['input'] if index == 0 else graph[layer_names[index-1]]
        if "conv" in layer_name:
            raw_layer = layer_dict[layer_name]
            graph[layer_name] = get_conv_layer(raw_layer, prev_layer)
        elif "relu" in layer_name:
            #apply relu but keep previous name
            graph[layer_name] = tf.nn.relu(prev_layer)        
        elif "pool" in layer_name:
            #apply pool but keep previous name
            graph[layer_name] = tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        else:
            #do nothing for now, we wont use the fully connected layers
            break
    return graph
    
