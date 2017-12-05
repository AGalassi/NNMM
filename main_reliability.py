#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2016-2017, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"

import sys

from testing import test_networks_reliability

def main(datasetname='DATASET.mock.txt',
         expanded=False,
         name='TEST-rawest-TFR'
         ):
    
    print("Testing dataset " + datasetname + "\nexpanded = " + str(expanded) +
          "\nnetworks: " + name)
    test_networks_reliability(datasetname, expanded, name)
    
    
if __name__ == '__main__':
    kwargs = {}
    usage = ("Usage: %s datasetname expanded netname" % sys.argv[0])
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Neural Nine Men's Morris\n" +
              "Reliability test mode:\n" +
              "Specify the dataset name, " +
              "if the dataset is already exapanded and " +
              "the name of the networks.")
        
    if len(sys.argv) == 4 :
        kwargs['datasetname'] = sys.argv[1]
        expanded = sys.argv[2]
        if (expanded == "False" or expanded =="false" or
            expanded == "FALSE"):
            kwargs['expanded'] = False
        elif (expanded == "True" or expanded =="true" or
              expanded == "TRUE"):
            kwargs['expanded'] = True
        else:
            print usage
            sys.exit(0)
        kwargs['name'] = sys.argv[3]
        main(**kwargs)
    elif len(sys.argv) == 1 :
        main(**kwargs)
    else:
        print ("Wrong number of arguments: " + str(len(sys.argv))+"\n" + usage)
