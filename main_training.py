#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2016-2017, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"

import sys
import lasagne.regularization as rgl

from training import train


def main():
    train(name='TEST-TFR', datasetname="dataset", expanded=True,
             vset_size=0.05, tset_size=0.1,
             movepart="TO",
             order = "TFR",
             batch_size=20000, num_epochs=10000,
             patience=50,
             nettype=2,
             neurons=[200, 300],
             blocks=10,
             lr_alfa0=0.002, b1=0.99, b2=0.999,
             lr_annealing=True,
             lr_k=0.01,
             dropi=0.1, drop=0.1,
             regularization=True,
             reg_type=rgl.l1,
             reg_weight=0.001,
             normalization=False,
             load=False, initial_epoch=0,
             data_format="binary rawer")
    
    
if __name__ == '__main__':
    kwargs = {}
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Neural Nine Men's Morris\n" +
              "Training of the networks. Change the parameters in this file" +
              "to configure training")
    elif len(sys.argv) == 1 :
        main(**kwargs)

