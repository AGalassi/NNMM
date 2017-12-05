#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2016-2017, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"

import sys

from testing import test_networks

def main(datasetname='DATASET.mock.txt',
         statesonly=False,
         expanded=False,
         name='TEST-rawest-TFR'
         ):
    
    print("Loading dataset " + datasetname + "\nstatesonly = " +
          str(statesonly) + "\nexpanded = " + str(expanded) +
          "\nnetworks: " + name)
    test_networks(datasetname, statesonly, expanded, name)
    
    
if __name__ == '__main__':
    kwargs = {}
    usage = ("Usage: %s datasetname statesonly expanded netname" % sys.argv[0])
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Neural Nine Men's Morris\n" +
              "Legality and precision test mode:\n" +
              "Specify the dataset name, " + 
              "if the dataset is composed only by states (no move parts), " +
              "if the dataset is already exapanded and " +
              "the name of the networks.")
    if len(sys.argv) == 5:
        kwargs['datasetname'] = sys.argv[1]
        
        statesonly = sys.argv[2]
        if (statesonly == "False" or statesonly =="false" or
            statesonly == "FALSE"):
            kwargs['statesonly'] = False
        elif (statesonly == "True" or statesonly =="true" or
              statesonly == "TRUE"):
            kwargs['statesonly'] = True
        else:
            print usage
            sys.exit(0)
        
        expanded = sys.argv[3]
        if (expanded == "False" or expanded =="false" or
            expanded == "FALSE"):
            kwargs['expanded'] = False
        elif (expanded == "True" or expanded =="true" or
              expanded == "TRUE"):
            kwargs['expanded'] = True
        else:
            print usage
            sys.exit(0)
        
        kwargs['name'] = sys.argv[4]
        main(**kwargs)
    elif len(sys.argv) == 1:
        main(**kwargs)
    else:
        print ("Wrong number of arguments: " + str(len(sys.argv))+"\n" + usage)
        

