#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2016-2017, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"

import numpy
import theano
import theano.tensor as T
import time
import lasagne
import sys
import lasagne.regularization as rgl
import pprint

from network import (load_net, get_predictions, get_choices)
from dataprocessing import (load_expanded_states_dataset, load_states_dataset,
                            process_state_binary, add_CHOICE_binary_raw,
                            load_expanded_dataset, load_dataset,
                            process_move_onlyTO, process_move_onlyFROM,
                            process_move_onlyREMOVE,)
from legality import get_legalities, near, is_phase_2, is_phase_1, is_phase_3


def test_networks_reliability(datasetname, expanded, name,
                              batchsize=10000):
    """
    Loads three existent networks and evaluate their mean reliability.

    The reliability is similar to a precision-recall curve, with the aim to
    measure the ability of a network to assign highest probability to legal
    moves.
    Defined the task to retrieve all the legal moves, the possible choices of
    each network are ranked accordingly their probability score.
    For defined percentages of recall, the precision percentage is computed.
    These values are saved in a realtive file.
    If the previous networks have taken an illegal choice for a state, that
    state it is not taken into account to compute the mean values.
    Also the cases in which the only legal choice is 0 are not considered.

    Parameters
    ----------
    datasetname : string
        The name of the dataset file.
    expanded : boolean
        True if the dataset has been already expanded through symmetries.
    name : string
        The name of the network configuration to load
    """
    
    start_time = time.time()
   
    print("Testing " + name)
    print("Loading networks...")
    print("\tloading " + name + "_TO")
    TOnet = load_net(name + "_TO")
    print("\tloading " + name + "_FROM")
    FROMnet = load_net(name + "_FROM") 

    print("\tloading " + name + "_REMOVE")
    REMOVEnet = load_net(name + "_REMOVE")
    print("\tNetworks loaded!")
    
    data_format = TOnet[3]
    
    # an array in which is loaded the configuration: TFR/RFT/FTR ecc.
    orderc = ['X', 'X', 'X']
    orderc[TOnet[2]] = 'T'
    orderc[FROMnet[2]] = 'F'
    orderc[REMOVEnet[2]] = 'R'
    order = "" + orderc[0] + orderc[1] + orderc[2]
    
    
    nets = ['X', 'X', 'X']
    nets[TOnet[2]] = TOnet
    nets[FROMnet[2]] = FROMnet
    nets[REMOVEnet[2]] = REMOVEnet
        
    
    ordernames = ['X', 'X', 'X']
    ordernames[TOnet[2]] = "TO"
    ordernames[FROMnet[2]] = "FROM"
    ordernames[REMOVEnet[2]] = "REMOVE"
    
    
    print("\tOrder: " + order)
    
    print("Loading data from \"" + datasetname + "\" ...")
    if expanded:
        A, B = load_expanded_states_dataset(datasetname)
    else:
        A, B = load_states_dataset(datasetname)

    print("\tData loaded! Loaded: " + str(len(A)) + " data")
    
    
    numbatch = int(len(A) / batchsize) + 1
    
    T_p = []
    O_p = []
    Z_p = []
    
    num_data = len(A)
    perc = 0
    
    for i in range(1, numbatch + 1):

        if (i * batchsize <= len(A)):
            Ai = A[(i-1) * batchsize: i * batchsize]
            percentage = i*batchsize*100/(num_data*1.0)

        else:
            Ai = A[(i-1) * batchsize:]
            percentage = 100.0

        X_0 = process_state_binary(Ai, data_format)
        bin_state = process_state_binary(Ai, "binary raw")
        
        ZERO_choice = get_choices(nets[0], X_0)
        ZERO_predictions = get_predictions(nets[0], X_0)
        
        X_1 = add_CHOICE_binary_raw(X_0, ZERO_choice)
        
        ONE_choice = get_choices(nets[1], X_1)
        ONE_predictions = get_predictions(nets[1], X_1)
        
        X_2 = add_CHOICE_binary_raw(X_1, ONE_choice)
        
        # remove scorings
        TWO_predictions = get_predictions(nets[2], X_2)
        
        # for each state
        for j in range(len(TWO_predictions)):
            
            moves = find_legal_moves(Ai[j], bin_state[j], "binary raw")
            
            # approximated precision values
            casep = []
            # real precision values
            prec = []
            
            # choices of the first and second networks
            ZC = ZERO_choice[j]
            OC = ONE_choice[j]
            
            choices = [ZC, OC, -1]
            
            Tsp = TWO_predictions[j]
            
            
            num_pred = len(Tsp)
                
            # se la seconda rete pensa che sia il caso di rimuovere
            if (numpy.argmax(Tsp) != 0):
                
                l = 0
                
                # SPEED UP: exclude this state if the previous networks have
                # already made illegal decisions
                TOchoice =choices[TOnet[2]]
                FROMchoice =choices[FROMnet[2]]
                REMOVEchoice =choices[REMOVEnet[2]]
                if orderc[2] == 'T':
                    move=("F" + str(FROMchoice) + "R" + str(REMOVEchoice))
                elif orderc[2] == 'F':
                    move=("T" + str(TOchoice) + "R" + str(REMOVEchoice))
                elif orderc[2] == 'R':
                    move=("T" + str(TOchoice) + "F" + str(FROMchoice))
                
                if move in moves:
                    # per ogni possibile terza scelta
                    for k in range(num_pred):
                        TWO_choice = numpy.argmax(Tsp)
                        Tsp[TWO_choice] = -1
                        choices[2] = TWO_choice
                        
                        TOchoice =choices[TOnet[2]]
                        FROMchoice =choices[FROMnet[2]]
                        REMOVEchoice =choices[REMOVEnet[2]]
                        
                        move=("T" + str(TOchoice) + "F" + str(FROMchoice) +
                              "R" + str(REMOVEchoice))
                        
                        # each time a legal move is found,
                        # compute the precision
                        if move in moves:
                            l += 1
                            prec.append(l*1.0/(k+1.0)*100.0)
                    
                    # compute the approximated values
                    if (l > 0):
                        rec = 0
                        for k in range(num_pred):
                            # if the approximated recal il less than the
                            # real recall
                            if ((k+1.0)/num_pred) <= ((rec+1.0)/l):
                                # assign the next value of precision
                                casep.append(prec[rec])
                            else:
                                # look at the next recall value and
                                # assign its precision value
                                rec +=1
                                casep.append(prec[rec])
                        T_p.append(casep)
        
            
            # probabilities of this state for the second network
            casep = []
            prec = []
            
            choices = [ZC, -1, -1] 
            Osp = ONE_predictions[j]
            
            
            num_pred = len(Osp)
            
            l = 0
            
            # exclude this state if 
            if (numpy.argmax(Osp) != 0):
                
                # SPEED UP: exclude this state if the previous networks have
                # already made illegal decisions
                TOchoice =choices[TOnet[2]]
                FROMchoice =choices[FROMnet[2]]
                REMOVEchoice =choices[REMOVEnet[2]]
                if orderc[0] == 'T':
                    move=("T" + str(TOchoice))
                elif orderc[0] == 'F':
                    move=("F" + str(FROMchoice))
                elif orderc[0] == 'R':
                    move=("R" + str(REMOVEchoice))

                if move in moves:
                    # per ogni possibile seconda scelta
                    for k in range(num_pred):
                        OC = numpy.argmax(Osp)
                        Osp[OC] = -1
                        
                        choices[1] = OC
                        
                        #print("Choices: " + str(choices))
                        
                        TOchoice = choices[TOnet[2]]
                        FROMchoice = choices[FROMnet[2]]
                        REMOVEchoice = choices[REMOVEnet[2]]
                        
                        move = "ERROR"
                        
                        # create the move according to the lacking network (the 3)
                        if orderc[2] == 'T':
                            move=("F" + str(FROMchoice) + "R" + str(REMOVEchoice))
                        elif orderc[2] == 'F':
                            move=("T" + str(TOchoice) + "R" + str(REMOVEchoice))
                        elif orderc[2] == 'R':
                            move=("T" + str(TOchoice) + "F" + str(FROMchoice))
                        
                        if move in moves:
                            l += 1
                            prec.append(l*1.0/(k+1.0)*100.0)
                    
                    if (l > 0):
                        rec = 0
                        for k in range(num_pred):
                            #print (rec)
                            if ((k+1.0)/num_pred) <= ((rec+1.0)/l):
                                casep.append(prec[rec])
                            else:
                                rec +=1
                                casep.append(prec[rec])
                        #print(str(casep))
                        O_p.append(casep)
        
            
            # probabilities of this state for the first network
            casep = []
            prec = []
            choices = [-1, -1, -1] 
            
            Zsp = ZERO_predictions[j]

            num_pred = len(Zsp)

            # se la prima rete pensa che sia il caso di rimuovere
            if (numpy.argmax(Zsp) != 0):
                l = 0.0
                
                # per ogni possibile rimozione
                for k in range(num_pred):
                    ZC = numpy.argmax(Zsp)
                    Zsp[ZC] = -1
                    choices[0] = ZC
                    
                    TOchoice = choices[TOnet[2]]
                    FROMchoice = choices[FROMnet[2]]
                    REMOVEchoice = choices[REMOVEnet[2]]
                    
                    
                    move = "ERROR"
                    
                    # create the move according to the only network
                    if orderc[0] == 'T':
                        move=("T" + str(TOchoice))
                    elif orderc[0] == 'F':
                        move=("F" + str(FROMchoice))
                    elif orderc[0] == 'R':
                        move=("R" + str(REMOVEchoice))
                    
                    if move in moves:
                        l += 1
                        prec.append(l*1.0/(k+1.0)*100.0)
                
                if (l > 0):
                    rec = 0
                    for k in range(num_pred):
                        #print (rec)
                        if ((k+1.0)/num_pred) <= ((rec+1.0)/l):
                            casep.append(prec[rec])
                        else:
                            rec +=1
                            casep.append(prec[rec])
                    #print(str(casep))
                    Z_p.append(casep)
        
        
        act_time = time.time()
        # gives feedback about the amount of data processed
        while(percentage > perc):
            perc += 1
        if perc!=100:
            print(str(round(percentage,2)) + "%\t" + str(i*batchsize) +
                  " data processed\ttime passed: " + str(act_time-start_time))
    
    print("100%\t" + str(num_data) + " data processed\ttime passed: " +
          str(act_time-start_time))
    
    filerel = open(name + "_reliability.txt", 'a')
    filerel.write("Test on: " + datasetname)
    
    report = ""
    
    report += ("\n\n" +ordernames[2] + ":\n")
    if len(T_p) > 0:
        m = numpy.mean(T_p, 0)
        for num in m:
            report += (str(round(num,2))+"\t")
    
    
    report += ("\n\n" + ordernames[1] + ":\n")
    if len(O_p) > 0:
        m = numpy.mean(O_p, 0)     
        for num in m:
            report += (str(round(num,2))+"\t")
    
    
    report += ("\n\n" + ordernames[0] + ":\n")
    m = numpy.mean(Z_p, 0)   
    for num in m:
        report += (str(round(num,2))+"\t")
    
    report += ("\n\n\n\n\n")
    filerel.write(report)
    print(report)
    filerel.close()
    
    act_time = time.time()
    
    
    print("Time occurred: " + str(act_time-start_time))



# returns a set with all the legal complete and partial moves, as strings
# ordered as TFR
def find_legal_moves(state, bin_state, data_format):
    """
    Compute all the legal partial and complete moves for the given state

    

    Parameters
    ----------
    state : State
        The game state.
    bin_state : int[]
        The game state represented as an array
    data_format : string
        The representation format of the bin_state
    
    Returns
    -------
    set
        A set of strings with all the legal partial and complete moves
    """
    
    moves = set()
    
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
                moves.add("T"+str(Tpart)+"F"+str(Fpart)+"R"+str(Rpart))
                moves.add("T"+str(Tpart)+"R"+str(Rpart))
                moves.add("F"+str(Fpart)+"R"+str(Rpart))
                moves.add("T"+str(Tpart)+"F"+str(Fpart))
                moves.add("T"+str(Tpart))
                moves.add("F"+str(Fpart))
                moves.add("R"+str(Rpart))
                
            # if one remove is necessary, all the enemy positions are
            # candidates
            else:
                for REMOVE_position in enemy_pos:
                    Rpart = REMOVE_position + 1
                    legality = get_legalities([Tpart], [Fpart], [Rpart],
                                              [bin_state], data_format)
                    if (legality[7][0] == 1):
                        moves.add("T"+str(Tpart)+"F"+str(Fpart)+"R"+str(Rpart))
                        moves.add("T"+str(Tpart)+"R"+str(Rpart))
                        moves.add("F"+str(Fpart)+"R"+str(Rpart))
                        moves.add("T"+str(Tpart)+"F"+str(Fpart))
                        moves.add("T"+str(Tpart))
                        moves.add("F"+str(Fpart))
                        moves.add("R"+str(Rpart))
        
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
                    moves.add("T"+str(Tpart)+"F"+str(Fpart)+"R"+str(Rpart))
                    moves.add("T"+str(Tpart)+"R"+str(Rpart))
                    moves.add("F"+str(Fpart)+"R"+str(Rpart))
                    moves.add("T"+str(Tpart)+"F"+str(Fpart))
                    moves.add("T"+str(Tpart))
                    moves.add("F"+str(Fpart))
                    moves.add("R"+str(Rpart))
                    
                # if the the legality of the FT couple is confirmed,
                # one remove is necessary, all the enemy positions are
                # candidates
                elif (legality[3][0] == 1):
                    for REMOVE_position in enemy_pos:
                        Rpart = REMOVE_position + 1
                        legality = get_legalities([Tpart], [Fpart], [Rpart],
                                                  [bin_state], data_format)
                        if (legality[7][0] == 1):
                            moves.add("T"+str(Tpart)+
                                      "F"+str(Fpart)+"R"+str(Rpart))
                            moves.add("T"+str(Tpart)+"R"+str(Rpart))
                            moves.add("F"+str(Fpart)+"R"+str(Rpart))
                            moves.add("T"+str(Tpart)+"F"+str(Fpart))
                            moves.add("T"+str(Tpart))
                            moves.add("F"+str(Fpart))
                            moves.add("R"+str(Rpart))
        
        
        
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
                    moves.add("T"+str(Tpart)+"F"+str(Fpart)+"R"+str(Rpart))
                    moves.add("T"+str(Tpart)+"R"+str(Rpart))
                    moves.add("F"+str(Fpart)+"R"+str(Rpart))
                    moves.add("T"+str(Tpart)+"F"+str(Fpart))
                    moves.add("T"+str(Tpart))
                    moves.add("F"+str(Fpart))
                    moves.add("R"+str(Rpart))
                    
                # if one remove is necessary, all the enemy positions are
                # candidates
                else:
                    for REMOVE_position in enemy_pos:
                        Rpart = REMOVE_position + 1
                        legality = get_legalities([Tpart], [Fpart], [Rpart],
                                                  [bin_state], data_format)
                        if (legality[7][0] == 1):
                            moves.add("T"+str(Tpart)+
                                      "F"+str(Fpart)+"R"+str(Rpart))
                            moves.add("T"+str(Tpart)+"R"+str(Rpart))
                            moves.add("F"+str(Fpart)+"R"+str(Rpart))
                            moves.add("T"+str(Tpart)+"F"+str(Fpart))
                            moves.add("T"+str(Tpart))
                            moves.add("F"+str(Fpart))
                            moves.add("R"+str(Rpart))


    return moves


# load three existent networks and use them to predict the moves on the given
# dataset. Then verify accuracy and legality of the predicted moves
def test_networks(datasetname, statesonly, expanded, name,
                  batchsize=20000):
    """
    Measure the accuracy and the legality of a triplet of networks
    
    Loads three networks and a dataset and than measure how accurate are the
    networks predictions with respect to the dataset and if their choices are
    legal. If the dataset is made only by states (therefore no moves), the
    accuracy test is not performed.

    Parameters
    ----------
    datasetname : string
        The name of the dataset file.
    statesonly : boolean
        True if the dataset has doesn't contains informations about the moves.
    expanded : boolean
        True if the dataset has been already expanded through symmetries.
    name : string
        The name of the network configuration to load
    """
    
    print("Testing " + name)
    print("Loading networks...")
    print("\tloading " + name + "_TO")
    TOnet = load_net(name + "_TO")
    print("\tloading " + name + "_FROM")
    FROMnet = load_net(name + "_FROM") 

    print("\tloading " + name + "_REMOVE")
    REMOVEnet = load_net(name + "_REMOVE")
    print("\tNetworks loaded!")
    
    data_format = TOnet[3]
    print(str(data_format))
    
    orderc = ['X', 'X', 'X']
    orderc[TOnet[2]] = 'T'
    orderc[FROMnet[2]] = 'F'
    orderc[REMOVEnet[2]] = 'R'
    order = "" + orderc[0] + orderc[1] + orderc[2]
    print("\tOrder: " + order)
    
    print("Loading data...")
    
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

    print("\tData loaded! Loaded: " + str(len(B)) + " data")

    print("Processing data and getting choices")
    X_TO_set = []
    X_FROM_set = []
    X_REMOVE_set = []

    TO_choice_set = []
    FROM_choice_set = []
    REMOVE_choice_set = []
    
    y_TO_set = []
    y_FROM_set = []
    y_REMOVE_set = []
    
    numbatch = int(len(A) / batchsize) + 1

    for i in range(1, numbatch + 1):
        print("\t" + str(i * batchsize * 100.0 / len(A)) + "%")

        if (i * batchsize <= len(A)):
            Ai = A[(i-1) * batchsize: i * batchsize]
            Bi = B[(i-1) * batchsize: i * batchsize]

        else:
            Ai = A[(i-1) * batchsize:]
            Bi = B[(i-1) * batchsize:]

        y_TO = process_move_onlyTO(Bi)
        y_FROM = process_move_onlyFROM(Bi)
        y_REMOVE = process_move_onlyREMOVE(Bi)
        
        init_state = process_state_binary(Ai, data_format)

        if order == "FTR":
            X_FROM = init_state
            FROM_choice = get_choices(FROMnet, X_FROM)
            
            X_TO = add_CHOICE_binary_raw(X_FROM, FROM_choice)
            TO_choice = get_choices(TOnet, X_TO)
            
            X_REMOVE = add_CHOICE_binary_raw(X_TO, TO_choice)
            REMOVE_choice = get_choices(REMOVEnet, X_REMOVE)
        elif order == "RFT":
            X_REMOVE = init_state
            REMOVE_choice = get_choices(REMOVEnet, X_REMOVE)

            X_FROM = add_CHOICE_binary_raw(X_REMOVE, REMOVE_choice)
            FROM_choice = get_choices(FROMnet, X_FROM)
            
            X_TO = add_CHOICE_binary_raw(X_FROM, FROM_choice)
            TO_choice = get_choices(TOnet, X_TO)
        elif order == "FRT":
            X_FROM = init_state
            FROM_choice = get_choices(FROMnet, X_FROM)
            
            X_REMOVE = add_CHOICE_binary_raw(X_FROM, FROM_choice)
            REMOVE_choice = get_choices(REMOVEnet, X_REMOVE)
            
            X_TO = add_CHOICE_binary_raw(X_REMOVE, REMOVE_choice)
            TO_choice = get_choices(TOnet, X_TO)
        elif order == "RTF":
            X_REMOVE = init_state
            REMOVE_choice = get_choices(REMOVEnet, X_REMOVE)
            
            X_TO = add_CHOICE_binary_raw(X_REMOVE, REMOVE_choice)
            
            TO_choice = get_choices(TOnet, X_TO)
            
            X_FROM = add_CHOICE_binary_raw(X_TO, TO_choice)
            
            FROM_choice = get_choices(FROMnet, X_FROM)
        elif order == "TRF":
            X_TO = init_state
            TO_choice = get_choices(TOnet, X_TO)
            
            X_REMOVE = add_CHOICE_binary_raw(X_TO, TO_choice)
            REMOVE_choice = get_choices(REMOVEnet, X_REMOVE)
            
            X_FROM = add_CHOICE_binary_raw(X_REMOVE, REMOVE_choice)
            FROM_choice = get_choices(FROMnet, X_FROM)
        elif order == "TFR":
            X_TO = init_state
            TO_choice = get_choices(TOnet, X_TO)
            
            #print("\tGot TO choices")
            
            X_FROM = add_CHOICE_binary_raw(X_TO, TO_choice)
            
            FROM_choice = get_choices(FROMnet, X_FROM)
            
            #print("\tGot FROM choices")
            
            X_REMOVE = add_CHOICE_binary_raw(X_FROM, FROM_choice)
            
            REMOVE_choice = get_choices(REMOVEnet, X_REMOVE)
        else:
            print("Unknown configuration. Using TFR")
            X_TO = init_state
            TO_choice = get_choices(TOnet, X_TO)
            
            #print("\tGot TO choices")
            
            X_FROM = add_CHOICE_binary_raw(X_TO, TO_choice)
            
            FROM_choice = get_choices(FROMnet, X_FROM)
            
            #print("\tGot FROM choices")
            
            X_REMOVE = add_CHOICE_binary_raw(X_FROM, FROM_choice)
            
            REMOVE_choice = get_choices(REMOVEnet, X_REMOVE)
    
        
        for j in range (len(X_TO)):
            X_TO_set.append(X_TO[j])
            X_FROM_set.append(X_FROM[j])
            X_REMOVE_set.append(X_REMOVE[j])
            
            TO_choice_set.append(TO_choice[j])
            FROM_choice_set.append(FROM_choice[j])
            REMOVE_choice_set.append(REMOVE_choice[j])
            
            y_TO_set.append(y_TO[j])
            y_FROM_set.append(y_FROM[j])
            y_REMOVE_set.append(y_REMOVE[j])
        
        #print("\tGot REMOVE choices")

    # variable renaming
    X_TO = X_TO_set
    X_FROM = X_FROM_set
    X_REMOVE = X_REMOVE_set

    TO_choice = TO_choice_set
    FROM_choice = FROM_choice_set
    REMOVE_choice = REMOVE_choice_set
    
    y_TO = y_TO_set
    y_FROM = y_FROM_set
    y_REMOVE = y_REMOVE_set
    
    print("Testing legality")
    
    legalities = get_legalities(TO_choice, FROM_choice,
                                REMOVE_choice, X_REMOVE, data_format)
    
    print("\ttesting the legality of " + str(len(legalities[0])) + " data\n")
    
    TO_self_leg = legalities[0]
    FROM_self_leg = legalities[1]
    REMOVE_self_leg = legalities[2]
    FROM_leg = legalities[3]
    REMOVE_leg = legalities[4]
    wholeFROM = legalities[5]
    wholeREMOVE = legalities[6]
    wholeMOVE = legalities[7]
    
    fileT = open(name + "_testing.txt", 'a')
    
    leg = ("Legality response on " + datasetname + ":" + 
           "\n\tOnly TO leg:\t" + str(numpy.mean(TO_self_leg) * 100) +
           "\n\tOnly FROM leg:\t" + str(numpy.mean(FROM_self_leg) * 100) +
           "\n\tOnly REMOVE leg:\t" + str(numpy.mean(REMOVE_self_leg) * 100) +
           "\n\tOnly FROM-TO leg:\t" + str(numpy.mean(FROM_leg) * 100) +
           "\n\tOnly REMOVE-FROM-TO leg:\t" + str(numpy.mean(REMOVE_leg) * 100) +
           "\n\tWhole FROM leg:\t" + str(numpy.mean(wholeFROM) * 100) +
           "\n\tWhole REMOVE leg:\t" + str(numpy.mean(wholeREMOVE) * 100) +
           "\n\tWhole MOVE leg:\t" + str(numpy.mean(wholeMOVE) * 100))
    
    print(leg)
    fileT.write(leg)
    
    print("\nTesting accuracy and special cases\n")
    
    # accuracy valutation and legality evaluation for some particular cases:
    # the FROM move in phase 2 and the REMOVE when is not 0
    correctTO = 0
    correctFROM = 0
    correctREMOVE = 0
    correctWHOLE = 0
    
    # legality for FROM in phase 2
    legalFROM2 = 0
    f2 = 0
    
    # legality for REMOVE != 0 for the network
    legalREMOVEeat = 0
    re = 0
    
    # accuracy when REMOVE != 0 in dataset
    correctREMOVEyes = 0
    ry = 0
    
    # accuracy when FROM != 0 in dataset
    correctFROMyes = 0
    fy = 0
    
    # accuracy for the different phases
    correctWHOLE1 = 0
    p1 = 0
    correctWHOLE2 = 0
    p2 = 0
    correctWHOLE3 = 0
    p3 = 0
    
    size = len(TO_choice)
    
    print("\tTesting the accuracy of " + str(size) + " data")
    
    for i in range (size):

        # TO DECISION
        if TO_choice[i] == y_TO[i]:
            correctTO += 1
            
        # legal FROM decision in phase 2
        if (is_phase_2(X_REMOVE[i], data_format)):
            f2 += 1
            # legality
            l = get_legalities(TO_choice[i:i+1], FROM_choice[i:i+1],
                               REMOVE_choice[i:i+1], X_REMOVE[i:i+1],
                               data_format)
            if (l[5] == 1):
                legalFROM2 += 1
        
        # FROM decision is different from 0
        if y_FROM[i] != 0:
            fy += 1
        
        # FROM decision
        if FROM_choice[i] == y_FROM[i]:
            correctFROM += 1
            # correct FROM decision if different from 0
            if y_FROM[i] != 0:
                correctFROMyes += 1
        
        # REMOVE decision when present
        if y_REMOVE[i] != 0:
            ry += 1
            
        if REMOVE_choice[i] != 0:
            re += 1
            # legality
            l = get_legalities(TO_choice[i:i+1], FROM_choice[i:i+1],
                               REMOVE_choice[i:i+1], X_REMOVE[i:i+1],
                               data_format)
            if (l[6] == 1):
                legalREMOVEeat += 1
                
        # REMOVE decision
        if REMOVE_choice[i] == y_REMOVE[i]:
            correctREMOVE += 1
            if (y_REMOVE[i] != 0):
                correctREMOVEyes += 1
        
        if is_phase_1(X_REMOVE[i], data_format):
            p1 += 1
        elif is_phase_3(X_REMOVE[i], data_format):
            p3 += 1
        else:
            p2 += 1
        
        # WHOLE move (in different phases of game)
        if (TO_choice[i] == y_TO[i] and FROM_choice[i] == y_FROM[i] and
                    REMOVE_choice[i] == y_REMOVE[i]) :
            correctWHOLE += 1
            
            if is_phase_1(X_REMOVE[i], data_format):
                correctWHOLE1 += 1
            elif is_phase_3(X_REMOVE[i], data_format):
                correctWHOLE3 += 1
            else:
                correctWHOLE2 += 1
            
    
    if (size > 0):
        correctTO = correctTO * 100.0 / size
        correctFROM = correctFROM * 100.0 / size
        correctREMOVE = correctREMOVE * 100.0 / size
        correctWHOLE = correctWHOLE * 100.0 / size
    else:
        correctTO = -1
        correctFROM = -1
        correctREMOVE = -1
        correctWHOLE = -1
    
    if (f2 > 0):   
        legalFROM2 = legalFROM2 * 100.0 / f2
    else:
        legalFROM2 = -1
        
    if (re > 0):
        legalREMOVEeat = (legalREMOVEeat * 100.0) / re
    else:
        legalREMOVEeat = -1
    
    if (ry > 0):
        correctREMOVEyes = correctREMOVEyes * 100.0 / ry
    else:
        correctREMOVEyes = -1
        
    if (fy > 0):
        correctFROMyes = correctFROMyes * 100.0 / fy
    else:
        correctFROMyes = -1
                
    if (p1 > 0):
        correctWHOLE1 = correctWHOLE1 * 100.0 / p1
    else:
        correctWHOLE1 = -1

    if (p2 > 0):
        correctWHOLE2 = correctWHOLE2 * 100.0 / p2
    else:
        correctWHOLE2 = -1
        
    if (p3 > 0):
        correctWHOLE3 = correctWHOLE3 * 100.0 / p3
    else:
        correctWHOLE3 = -1
    
    leg = ("\n\tFROM phase 2 leg:\t" + str(legalFROM2) +
           "\n\tREMOVE when chosen leg:\t" + str(legalREMOVEeat))
    
    if(not statesonly):
        leg += ("\n--------------------\n\nAccuracy response on " +
                datasetname + ":" +
                "\n\tTO accuracy:\t" + str(correctTO) +
                "\n\tFROM accuracy:\t" + str(correctFROM) +
                "\n\tREMOVE accuracy:\t" + str(correctREMOVE) +
                "\n\tWhole MOVE accuracy:\t" + str(correctWHOLE) +
                "\n\n\tWhole MOVE accuracy in phase 1:\t" + str(correctWHOLE1)+
                "\n\tWhole MOVE accuracy in phase 2:\t" + str(correctWHOLE2) +
                "\n\tWhole MOVE accuracy in phase 3:\t" + str(correctWHOLE3) +
                "\n\n\tFROM not 0 accuracy:\t" + str(correctFROMyes) +
                "\n\tREMOVE not 0 accuracy:\t" + str(correctREMOVEyes) +
                "\n\n\n")
    
    print(leg)
    fileT.write(leg)
    
    fileT.close()
    