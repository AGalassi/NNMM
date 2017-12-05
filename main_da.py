#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2016-2017, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"

import sys

from dataprocessing import *

def main(datasetname='DATASET.mock.txt',
         statesonly=False,
         expanded=False
         ):
    print("Loading dataset " + datasetname + "\nstatesonly = " +
          str(statesonly) + "\nexpanded = " + str(expanded))
    
    if statesonly:
        if(expanded):
            A, B = load_expanded_states_dataset(datasetname)
        else:
            A, B = load_states_dataset(datasetname)
    else:
        if expanded:
            A, B = load_expanded_dataset(datasetname)
        else:
            A, B = load_dataset(datasetname)
            
    yTO = process_move_onlyTO(B)
    fileDA = open(datasetname + "_da.txt", 'a')
    fileDA.write ("Analysis of dataset:\t" + datasetname)
    
    p1 = 0
    p2 = 0
    p3 = 0
    
    for state in A:
        if (state.my_phase == 1):
            p1 += 1
        elif (state.my_phase == 2):
            p2 += 1
        elif (state.my_phase == 3):
            p3 += 1
    fileDA.write("Phases:\n")
    s = "1: \t" + str(p1) + "\n2: \t" + str(p2) + "\n3: \t" + str(p3)
    print(s)
    fileDA.write(s)

    fileDA.write("\n\nTO:\n")
    dicTO = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
             10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0,
             19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0}
    for to in yTO:
        dicTO[to] += 1
    for i in range(25):
        fileDA.write(str(i) + "\t" + str(dicTO[i]) +"\n")
    
    print("TO done. Making FROM")
    yFROM = process_move_onlyFROM(B)
    fileDA.write("\n\nFROM:\n")
    
    dicFROM = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
               10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0,
               18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0}
    for to in yFROM:
        dicFROM[to] += 1
    for i in range(25):
        fileDA.write(str(i) + "\t" + str(dicFROM[i]) + "\n")
        
    print("FROM done. Making REMOVE")    
    yREMOVE = process_move_onlyREMOVE(B)
    fileDA.write("\n\nREMOVE:\n")
    
    dicREMOVE = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
                 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0,
                 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0}
    for to in yREMOVE:
        dicREMOVE[to] += 1
    for i in range(25):
        fileDA.write(str(i) + "\t" + str(dicREMOVE[i]) + "\n")
    
    fileDA.write("\n\n___________________________\n\n\n\n")
    fileDA.close()
    
    
if __name__ == '__main__':
    kwargs = {}
    usage = ("Usage: %s datasetname statesonly expanded" % sys.argv[0])
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Neural Nine Men's Morris\n" +
              "Analysis of the phases of the states and the suggested moves " +
              "of the dataset:\n" +
              "Specify the dataset name, " + 
              "if the dataset is composed only by states (no move parts), " +
              "if the dataset is already exapanded.")
        
    if len(sys.argv) == 4:
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
        
        main(**kwargs)
    elif len(sys.argv) == 1:
        main(**kwargs)
    else:
        print ("Wrong number of arguments: " + str(len(sys.argv))+"\n" + usage)

