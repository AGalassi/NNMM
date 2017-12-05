#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2016-2017, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"

import lasagne
import cPickle as pickle
import numpy
import theano
import theano.tensor as T


# creates a dense network
# each block is composed by:
#       WEIGHT, (BN), RELU, (DROP)
def build_densenet(input_var=None, input_size=114,
                   neurons=[200, 200],
                   blocks=1,
                   pi=0, p=0, norm=False, nout=24):
    """
    Creates a neural networks based on the densely connected model
    
    

    Parameters
    ----------
    input_var : Theano symbolic variable or None (default: None)
        A variable representing a network input.
    input_size : int
        The size of the input
    neurons : int[]
        The number of neurons of the first layer and of each other layer
    blocks : int
        The number of blocks of the networks
    pi : float in [0,1]
        The probability associated with the dropout of the initial layer
    p : float in [0,1]
        The probability associated with the dropout of the other layers
    norm : boolean
        Whether the batch normalization will be applied
    nout : int
        The size of the output
    
    Returns
    -------

    """

    l_in = lasagne.layers.InputLayer(shape=(None, input_size),
                                     input_var=input_var,
                                     name="inputL")
    if pi > 0:
        l_in = lasagne.layers.DropoutLayer(l_in, pi, name="dropin")

    hidd_1 = lasagne.layers.DenseLayer(
            l_in, num_units=neurons[0],
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            name="hidden0")

    prev_l = hidd_1

    # list of input for layer
    prev_b = []
    prev_b.append(prev_l)

    for i in range(0, blocks):

        # merge
        prev_l = lasagne.layers.ConcatLayer(prev_b, axis=1,
                                            name="merge" + str(i))

        # BN
        if norm:
            prev_l = lasagne.layers.BatchNormLayer(prev_l,
                                                   name="norm" + str(i))

        # ReLU
        prev_l = lasagne.layers.NonlinearityLayer(
                prev_l,
                nonlinearity=lasagne.nonlinearities.rectify,
                name="R" + str(i))

        # dropout
        if p > 0:
            prev_l = lasagne.layers.DropoutLayer(
                prev_l, p, name="drop" + str(i))

        # weight
        prev_l = lasagne.layers.DenseLayer(
            prev_l, num_units=neurons[1],
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            name="hidden" + str(i))

        prev_b.append(prev_l)

    # Output
    # 1° softmax: WhereToPut, 24 possible position
    l_out_1 = lasagne.layers.DenseLayer(
        prev_l, num_units=nout,
        nonlinearity=lasagne.nonlinearities.softmax,
        name="outputL")

    return l_out_1


# Creates a residual network as in "Identity Mapping in Deep ResNets"
# Each block is made by:
#    (BN), RELU, (DROP), WEIGHT, (BN), RELU, (DROP), WEIGHT
# Where put the dropout?
def build_resnet(input_var=None, input_size=114,
                 neurons=[200, 200],
                 blocks=3,
                 pi=0, p=0,
                 norm=False, nout=24):
    """
    Creates a neural networks based on the residual network model
    
    

    Parameters
    ----------
    input_var : Theano symbolic variable or None (default: None)
        A variable representing a network input.
    input_size : int
        The size of the input
    neurons : int[]
        The number of neurons of the first layer and of each other layer
    blocks : int
        The number of blocks of the networks
    pi : float in [0,1]
        The probability associated with the dropout of the initial layer
    p : float in [0,1]
        The probability associated with the dropout of the other layers
    norm : boolean
        Whether the batch normalization will be applied
    nout : int
        The size of the output
    
    Returns
    -------
        The network
    """

    # Primo parametro: dimensione batch, secondo: numero di input
    l_in = lasagne.layers.InputLayer(shape=(None, input_size),
                                     input_var=input_var,
                                     name="inputL")

    # forse fa esplodere tutto. nel caso è da togliere
    if pi > 0:
        l_in = lasagne.layers.DropoutLayer(l_in, pi, name="dropin")

    l_in = lasagne.layers.DenseLayer(
            l_in, num_units=neurons[0],
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            name="hidden00")

    prev_b = l_in

    for i in range(0, blocks):

        if norm:
            prev_l = lasagne.layers.BatchNormLayer(prev_b,
                                                   name="norm" + str(i) + "1")
        else:
            prev_l = prev_b

        prev_l = lasagne.layers.NonlinearityLayer(
                prev_l,
                nonlinearity=lasagne.nonlinearities.rectify,
                name="R" + str(i) + "1")

        if p > 0:
            prev_l = lasagne.layers.DropoutLayer(
                prev_l, p, name="hd" + str(i) + "1")

        prev_l = lasagne.layers.DenseLayer(
            prev_l, num_units=neurons[1],
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            name="hidden" + str(i) + "1")

        if norm:
            prev_l = lasagne.layers.BatchNormLayer(prev_l,
                                                   name="norm" + str(i) + "2")

        prev_l = lasagne.layers.NonlinearityLayer(
                prev_l,
                nonlinearity=lasagne.nonlinearities.rectify,
                name="R" + str(i) + "2")

        if p > 0:
            prev_l = lasagne.layers.DropoutLayer(
                prev_l, p, name="hd" + str(i) + "2")

        prev_l = lasagne.layers.DenseLayer(
            prev_l, num_units=neurons[0],
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            name="hidden" + str(i) + "2")

        # SUM
        block = lasagne.layers.NonlinearityLayer(
            lasagne.layers.ElemwiseSumLayer([prev_l, prev_b]),
            nonlinearity=None,
            name="block" + str(i))

        prev_l = block

    # Output
    # 1° softmax: WhereToPut, 24 possible position
    l_out_1 = lasagne.layers.DenseLayer(
        prev_l, num_units=nout,
        nonlinearity=lasagne.nonlinearities.softmax,
        name="outputL")

    return l_out_1


def build_ffnet(input_var=None, input_size=114,
                neurons=[200, 200, 100, 50],
                pi=0, p=0, norm=False, nout=24):
    """
    Builds a feed forward neural network.
    Parameters
    ----------
    input_var : Theano symbolic variable or None (default: None)
        A variable representing a network input.
    input_size : int32
        The size of an single feature array
    neurons : int[]
        The number of neurons for each layer
    pi : float32 in [0,1]
        The probability of the dropout on the input layer
    p : float32 in [0,1]
        The probability of the dropout on the hidden layers
    norm : boolean
        True if the batch normalization should be applied
    nout : int32
        The number of class of the output
    Returns
    -------
        The network
    """
    
    # Primo parametro: dimensione batch, secondo: numero di input
    l_in = lasagne.layers.InputLayer(shape=(None, input_size),
                                     input_var=input_var,
                                     name="inputL")

    if pi > 0:
        l_in = lasagne.layers.DropoutLayer(l_in, pi, name="dropin")

    blocks = len(neurons)

    prevl = l_in

    for i in range(0, blocks):

        prevl = lasagne.layers.DenseLayer(
            prevl, num_units=neurons[i],
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            name="hidden"+str(i))

        if norm:
            prevl = lasagne.layers.BatchNormLayer(prevl, name="norm"+str(i))

        prevl = lasagne.layers.NonlinearityLayer(
            prevl,
            nonlinearity=lasagne.nonlinearities.rectify,
            name="R"+str(i))

        if p > 0:
            # We'll now add dropout of p%:
            prevl = lasagne.layers.DropoutLayer(prevl, p, name="drop"+str(i))

    # Output
    # 1° softmax: WhereToPut, 24 possible position
    l_out_1 = lasagne.layers.DenseLayer(
        prevl, num_units=nout,
        nonlinearity=lasagne.nonlinearities.softmax,
        name="outputL")

    # if (mask is not None):
    #    l_out_1 = l_out_1 * mask

    return l_out_1


def save_net_weights(network, name):
    """
    Saves a network
    
    Save all the parameters of a neural networks as file with extension *.npz

    Parameters
    ----------
    network : 
        The network to be saved
    name : string
        The name of the file
        
    """
    numpy.savez(name + '.npz',
                *lasagne.layers.get_all_param_values(network))


def load_net_weights(network, name):
    """
    Loads a network parameters
    
    Load the network parameters from a file *.npz and sets them into
    the provided network
    
    Parameters
    ----------
    network : 
        The network into which the parameters will be setted
    name : string
        The name of the file
        
    """

    with numpy.load(name + '.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    return network

def load_net(name):
    """
    Loads a network
    
    Loads the network, according to a file *.txt which defines the model, and
    a file *.npz which defines the parameters of the network
    
    Parameters
    ----------
    name : string
        The name of the files (without extension)
    
    Returns
    -------
    Theano symbolic variable or None
        A variable representing the network input.
    Neural network
        The loaded neural network
    int in [0,2]
        The position of the network in the cascade model
    string
        The representation format that must be used for the input data
    """

    # carico il dataset
    lines = open(name + ".txt", 'r').read().splitlines()
    #lines = open(name + "_net" + ".txt", 'r').read().splitlines()
    
    input_var = T.imatrix('inputs')
    nettype = -1
    neurons = [None] * 2
    blocks = -1
    nin = -1
    nout = 25
    batchnorm = False
    
    i = 0
    
    
    while "net" not in lines[i]:
        i += 1
    
    line = lines[i]
    
    if "ffnet" in line:
        nettype = 1
        sublines = line.split('\t')
        subline = sublines[1][1:]
        subline = subline[:-1]
        sublines = subline.split(", ")
        l = len (sublines)
        neurons = [None] * l
        for j in range(l):
            neurons[j] = int(sublines[j])
    elif "resnet" in line:
        nettype = 2
        sublines = line.split('\t')
        sublines = sublines[1].split()
        neurons = [None] * 2
        neurons[0] = int(sublines[0][:-1])
        neurons[1] = int(sublines[3])
        blocks = int(sublines[1])
        
    elif "densenet" in line:
        nettype = 3
        sublines = line.split('\t')
        sublines = sublines[1].split()
        neurons = [None] * 2
        neurons[0] = int(sublines[0][:-1])
        neurons[1] = int(sublines[3])
        blocks = int(sublines[1])
    else:
        exit()
    
    order = ""
    data_format = ""
    num = -1
    
    while "input" not in lines[i]:
        i += 1
    
    
    line = lines[i]
    
    
    if "binary rawer" in line:
        data_format = "binary rawer"
    elif "binary rawest" in line:
        data_format = "binary rawest"
    elif "binary raw" in line:
        data_format = "binary raw"
    
    if "FTR" in line:
        order = "FTR"
    elif "FRT" in line:
        order = "FRT"
    elif "TFR" in line:
        order = "TFR"
    elif "TRF" in line:
        order = "TRF"
    elif "RFT" in line:
        order = "RFT"
    elif "RTF" in line:
        order = "RTF"
    else:
        order = "TFR"

    for s in line.split('\t'):
        if s.isdigit():
            nin = int(s)
            break

    while "output" not in lines[i]:
        i += 1
    
    line = lines[i]
    
    if "TO" in line:
        num = order.index('T')
        if (nin == -1):
            nin = 114
    elif "FROM" in line:
        num = order.index('F')
        if (nin == -1):
            nin = 139
    elif "REMOVE" in line:
        num = order.index('R')
        if (nin == -1):
            nin = 164
    
    for s in line.split():
        if s.isdigit():
            nout = int(s)
    
    while "batch normalization" not in lines[i]:
        i += 1
        
    if ("yes" in lines[i]) or ("true" in lines[i]):
        batchnorm = True
    
    if (nettype == 1):
        network = build_ffnet(input_var=input_var,
                              input_size=nin,
                              neurons=neurons,
                              pi=0, p=0,
                              norm=batchnorm,
                              nout=nout)
        
    elif (nettype == 2):
        network = build_resnet(input_var=input_var,
                               input_size=nin,
                               neurons=neurons,
                               blocks=blocks,
                               pi=0, p=0,
                               norm=batchnorm,
                               nout=nout)

    elif (nettype == 3):
        network = build_densenet(input_var=input_var,
                               input_size=nin,
                               neurons=neurons,
                               blocks=blocks,
                               pi=0, p=0,
                               norm=batchnorm,
                               nout=nout)
        
    load_net_weights(network, name)
    
    return input_var, network, num, data_format


# input:
#   network: array with the input in position 0 and the network in position 1
#   X_test: the processed input data
# output:
#   for each case in X_test, the better choice
def get_choices(network, X_test):
    """
    The network evaluates the data and provides the best choices for each data
    
    For each data provided as input, the network computes which is the best
    choice and provide them as output
    
    Parameters
    ----------
    network: array
        The array has input of the network in position 0 and
        the network itself in position 1
    X_test: int[][]
        The processed input data
    
    Returns
    -------
    int[]
        The best choice for each input
    """
    net = network[1]
    input_var = network[0]
    test_prediction = lasagne.layers.get_output(net, deterministic=True)
    test_choice = T.argmax(test_prediction, axis=1)
    test_choice_fn = theano.function(
        [input_var], test_choice, name="test choice function")

    choices = test_choice_fn(X_test)
    return choices


# input:
#   network: array with the input in position 0 and the network in position 1
#   X_test: the processed input data
# output:
#   for each case in X_test, the probability of all the choices
def get_predictions(network, X_test):
    """
    The network evaluates the data and provides the probability scores for
    each possibility for each data
    
    For each data provided as input, the network computes which are the
    probability scores associated with each data
    
    Parameters
    ----------
    network: array
        The array has input of the network in position 0 and
        the network itself in position 1
    X_test: int[][]
        The processed input data
    
    Returns
    -------
    int[][]
        The probability scores for each input
    """
    net = network[1]
    input_var = network[0]
    test_prediction = lasagne.layers.get_output(net, deterministic=True)
    test_prediction_fn = theano.function(
        [input_var], test_prediction, name="test prediction function")

    choices = test_prediction_fn(X_test)
    return choices

