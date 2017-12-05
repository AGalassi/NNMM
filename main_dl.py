#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2016-2017, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"

import sys
import time
import numpy

from dataprocessing import (load_expanded_states_dataset, load_states_dataset,
                            process_state_binary)
from legality import get_legalities, near

# load the states of a database and measures the number of legal moves for each
# state and the number of moves in which each move is legal
def main(datasetname='DATASET.mock.txt',
         expanded=False,
         ):
    """
    Computes statistics relative to a dataset legality.
    
    Loads a dataset and measures the mean number of legal moves possible in
    each state and thenumber of states in which a particular move is legal.

    Parameters
    ----------
    datasetname : string
        The name of the dataset file.
    expanded : boolean
        True if the dataset has been already expanded through symmetries.
    """
    
    
    print("Testing " + datasetname + "\nexpanded = " + str(expanded))
    
    print("Testing legality characteristics of " + datasetname +
          "\nLoading data...")
    
    if(expanded):
        A, B = load_expanded_states_dataset(datasetname)
    else:
        A, B = load_states_dataset(datasetname)
    
    data_format = "binary raw"
    X_train = process_state_binary(A, data_format)
    
    n_states = len(A)
    
    TO_moves = range(1,25)
    FROM_moves = range(0,25)
    REMOVE_moves = range(0,25)
    
    n_moves = len(TO_moves)*len(FROM_moves)*len(REMOVE_moves)
    
    # counter of number of legal moves for each states
    count_states = [0] * n_states
    
    # counter of states in which this move is legal
    count_moves = [0] * n_moves
    
    print(str(n_states) + " data loaded. " + str(n_moves) + " moves considered")
    
    fileStates = open(datasetname + "_legmeas_states.txt", 'w')
    fileMoves = open(datasetname + "_legmeas_moves.txt", 'w')
    fileRecap = open(datasetname + "_legmeas.txt", 'w')
    
    fileStates.write("Testing legality characteristics of " + datasetname + "\n")
    fileMoves.write("Testing legality characteristics of " + datasetname + "\n")
    fileMoves.write("T\tF\tR\n")
    fileRecap.write("Testing legality characteristics of " + datasetname + "\n")
    
    perc=1
    start_time = time.time()
    print("Starting the test...")
    
    for i in range(n_states):
        bin_state = X_train[i]
        state = A[i]
        
        # build 3 set: empty positions, positions with enemy checkers and
        # positions with owned checkers
        enemy_pos = []
        mine_pos = []
        empty_pos = []
        for position in range(24):
            checker = state.positions[position]
            if checker == 'M':
                mine_pos.append(position)
            elif checker == 'E':
                enemy_pos.append(position)
            elif checker == 'O':
                empty_pos.append(position)
        
        Tpart = -1
        Fpart = -1
        Rpart = -1
        
        if (state.my_phase == 1):
            
            # the only legal FROM is 0
            Fpart = 0
            
            # the legal TO positions are the empty ones
            for TO_position in empty_pos:
                Tpart = TO_position + 1
                
                # verify if the option of no remove is legal
                Rpart = 0
                
                legality = get_legalities([Tpart], [Fpart], [Rpart],
                                          [bin_state], data_format)
                if (legality[7][0] == 1):
                    count_states[i] += 1
                    index = (((TO_position) * len(REMOVE_moves) + Fpart) *
                             len(FROM_moves) + Rpart)
                    count_moves[index] += 1
                    
                # if one remove is necessary, all the enemy positions are
                # candidates
                else:
                    for REMOVE_position in enemy_pos:
                        Rpart = REMOVE_position + 1
                        legality = get_legalities([Tpart], [Fpart], [Rpart],
                                                  [bin_state], data_format)
                        if (legality[7][0] == 1):
                            count_states[i] += 1
                            index = (((TO_position) * len(REMOVE_moves) +
                                      Fpart) * len(FROM_moves) + Rpart)
                            count_moves[index] += 1
            
        elif (state.my_phase == 2):
            # each checker is a legal from position
            for FROM_position in mine_pos:
                Fpart = FROM_position + 1
            
                # find the positions which are adjacents to the FROM one
                adjacent_parts = []
                
                for part in range(len(near[Fpart])):
                    if (near[Fpart][part] == 1):
                        adjacent_parts.append(part)
                
                for Tpart in adjacent_parts:
                    TO_position = Tpart - 1
                    
                    # verify if the option of no remove is legal
                    Rpart = 0
                        
                    legality = get_legalities([Tpart], [Fpart], [Rpart],
                                              [bin_state], data_format)
                    if (legality[7][0] == 1):
                        count_states[i] += 1
                        index = (((TO_position) * len(REMOVE_moves) + Fpart) *
                                 len(FROM_moves) + Rpart)
                        count_moves[index] += 1
                        
                    # if the the legality of the FT couple is confirmed,
                    # one remove is necessary, all the enemy positions are
                    # candidates
                    elif (legality[3][0] == 1):
                        for REMOVE_position in enemy_pos:
                            Rpart = REMOVE_position + 1
                            legality = get_legalities([Tpart], [Fpart], [Rpart],
                                                      [bin_state], data_format)
                            if (legality[7][0] == 1):
                                count_states[i] += 1
                                index = (((TO_position) * len(REMOVE_moves) +
                                          Fpart) * len(FROM_moves) + Rpart)
                                count_moves[index] += 1
            
            
            
        elif (state.my_phase == 3):
            # each empty position is legal
            for TO_position in empty_pos:
                Tpart = TO_position + 1
                # each checker is a legal from position
                for FROM_position in mine_pos:
                    Fpart = FROM_position + 1
                    
                    # verify if the option of no remove is legal
                    Rpart = 0
                    
                    legality = get_legalities([Tpart], [Fpart], [Rpart],
                                              [bin_state], data_format)
                    if (legality[7][0] == 1):
                        count_states[i] += 1
                        index = (((TO_position) * len(REMOVE_moves) + Fpart) *
                                 len(FROM_moves) + Rpart)
                        count_moves[index] += 1
                        
                    # if one remove is necessary, all the enemy positions are
                    # candidates
                    else:
                        for REMOVE_position in enemy_pos:
                            Rpart = REMOVE_position + 1
                            legality = get_legalities([Tpart], [Fpart], [Rpart],
                                                      [bin_state], data_format)
                            if (legality[7][0] == 1):
                                count_states[i] += 1
                                index = (((TO_position) * len(REMOVE_moves) +
                                          Fpart) * len(FROM_moves) + Rpart)
                                count_moves[index] += 1
                        
                        
        fileStates.write(str(A[i]) + "\t-\t" + str(count_states[i]) + "\n")
        
        
        act_time = time.time()
        percentage = i*100/(n_states*1.0)
        # gives feedback about the amount of data processed
        while(percentage > perc):
            perc += 1
        print(str(round(percentage,2)) + "%\t" + str(i) +
              " data processed\ttime passed: " + str(act_time-start_time))
        fileStates.flush()
    
    act_time = time.time()
    print("Data processed. Seconds passed: " + str(act_time-start_time))
    
    fileStates.close()
    
    for TO_opt in TO_moves:
        for FROM_opt in FROM_moves:
            for REMOVE_opt in REMOVE_moves:
                index = ((TO_opt-1) * len(FROM_moves) * len(REMOVE_moves) +
                         FROM_opt * len(REMOVE_moves) + REMOVE_opt)
                num = count_moves[index]
                fileMoves.write(str(TO_opt) + "\t" + str(FROM_opt) + "\t" +
                                str(REMOVE_opt) + "\t-\t" + str(num) + "\n")
    
    fileMoves.close()
    
    meanStates = numpy.mean(count_states)
    meanMoves = numpy.mean(count_moves)
    fileRecap.write("Mean legal moves per state:\t" + str(meanStates) + "\n")
    fileRecap.write("Mean states in which the move is legal:\t" +
                    str(meanMoves) + "\n\n")
    
    table = {}
    for count in count_states:
        # count the number of moves as a percentage of the total
        percentage = count
        if percentage in table.keys():
            table[percentage] += 1
        else:
            table[percentage] = 1
    
    for percentage in table.keys():
        fileRecap.write(str(percentage) + "\t" + str(table[percentage]) + "\n")
    
    fileRecap.close()            
    
    

    
    
if __name__ == '__main__':
    kwargs = {}
    usage = ("Usage: %s datasetname expanded" % sys.argv[0])
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Neural Nine Men's Morris\n" +
              "Analysis of the legality characteristics of the dataset:\n" +
              "Specify the dataset name and " + 
              "if the dataset is already exapanded")
    if len(sys.argv) == 3:
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
        
        main(**kwargs)
    elif len(sys.argv) == 1 :
        main(**kwargs)
    else:
        print ("Wrong number of arguments: " + str(len(sys.argv))+"\n" + usage)

