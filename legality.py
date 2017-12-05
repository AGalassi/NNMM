#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2016-2017, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"

import numpy

# position which are aligned on the same line of the key
# in particular, the left-most position
row1 = {
        0: 0,
        1: 2,
        2: 1,
        3: 1,
        4: 5,
        5: 4,
        6: 4,
        7: 8,
        8: 7,
        9: 7,
        10: 11,
        11: 10,
        12: 10,
        13: 14,
        14: 13,
        15: 13,
        16: 17,
        17: 16,
        18: 16,
        19: 20,
        20: 19,
        21: 19,
        22: 23,
        23: 22,
        24: 22
        }

# position which are aligned on the same line of the key
# in particular, the right-most position
row2 = {
        0: 0,
        1: 3,
        2: 3,
        3: 2,
        4: 6,
        5: 6,
        6: 5,
        7: 9,
        8: 9,
        9: 8,
        10: 12,
        11: 12,
        12: 11,
        13: 15,
        14: 15,
        15: 14,
        16: 18,
        17: 18,
        18: 17,
        19: 21,
        20: 21,
        21: 20,
        22: 24,
        23: 24,
        24: 23
        }

# position which are aligned on the same column of the key
# in particular, the up-most position
column1 = {
           0: 0,
           1: 10,
           2: 5,
           3: 15,
           4: 11,
           5: 2,
           6: 14,
           7: 12,
           8: 2,
           9: 13,
           10: 1,
           11: 4,
           12: 7,
           13: 9,
           14: 6,
           15: 3,
           16: 7,
           17: 20,
           18: 9,
           19: 4,
           20: 17,
           21: 6,
           22: 1,
           23: 17,
           24: 3
        }

# position which are aligned on the same column of the key
# in particular, the down-most position
column2 = {
           0: 0,
           1: 22,
           2: 8,
           3: 24,
           4: 19,
           5: 8,
           6: 21,
           7: 16,
           8: 5,
           9: 18,
           10: 22,
           11: 19,
           12: 16,
           13: 18,
           14: 21,
           15: 24,
           16: 12,
           17: 23,
           18: 13,
           19: 11,
           20: 23,
           21: 14,
           22: 10,
           23: 20,
           24: 15
        }

# for each position (1-24) indicates which of the other 24 positions (1-24)
# are adjacent
near = {
        0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        1: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        2: [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        3: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        4: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        5: [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        6: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        8: [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        9: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0],
        10: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0],
        11: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0],
        12: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 0, 0],
        13: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0],
        14: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 0],
        15: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1],
        16: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 0],
        17: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
             0, 0, 0, 0],
        18: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 0],
        19: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0],
        20: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
             1, 0, 1, 0],
        21: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0],
        22: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 0],
        23: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
             0, 1, 0, 1],
        24: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 1, 0]
        }


def TFR_legality_test(choicesTO, choicesFROM, choicesREMOVE,
                         X_REMOVE, data_format="binary raw"):
    num = len(choicesTO)

    result = numpy.zeros((num), dtype="int32")
    for i in range(num):
        choiceTO = choicesTO[i]
        choiceFROM = choicesFROM[i]
        choiceREMOVE = choicesREMOVE[i]

        absence = 0
        if (choiceREMOVE == 0):
            absence = 1

        # find the other row/colum position
        opta = row1[choiceTO]
        optb = row2[choiceTO]
        optc = column1[choiceTO]
        optd = column2[choiceTO]

        trisrow = 0
        triscol = 0

        # verify if in the column/row there are the player's stones
        if opta != choiceFROM and optb != choiceFROM:
            p1 = is_position_player(X_REMOVE[i], opta, data_format)
            p2 = is_position_player(X_REMOVE[i], optb, data_format)
            if p1 and p2:
                trisrow = 1

        if optc != choiceFROM and optd != choiceFROM:
            p1 = is_position_player(X_REMOVE[i], optc, data_format)
            p2 = is_position_player(X_REMOVE[i], optd, data_format)
            if p1 and p2:
                triscol = 1

        # if there are three checkers aligned, in one direction or both, there
        # is a mill
        checktris = trisrow * (1 - triscol) + triscol * (1 - trisrow) +\
                    trisrow * triscol

        # if there is a tris, the remove choice must be present
        # if there is not a tris, the remove choice must be not present
        result[i] = checktris * (1 - absence) + absence * (1 - checktris)

    return result


def FT_legality_test(choicesTO, choicesFROM, X_FROM,
                       data_format="binary raw"):
    num = len(choicesTO)
    result = numpy.zeros((num), dtype="int32")
    for i in range(num):
        choiceTO = choicesTO[i]
        choiceFROM = choicesFROM[i]
        legal = 0
        # if the phase is 3, every position is legal
        if is_phase_3(X_FROM[i], data_format):
            legal = 1
        # if the choice is 0, is always good
        # (the legality of choice 0 is checked in the self-legality test)
        if choiceFROM == 0:
            legal = 1
        # otherwise, the position must be adiacent to the TO position
        cn = near[choiceFROM][choiceTO]
        result[i] = cn * (1 - legal) + legal
    return result


# check if the chosen position is empty
# in phase 2 check if there is a friendly stone in adjacent position
def TO_self_legality_test(choicesTO, X_TO, data_format="binary raw"):
    num = len(choicesTO)
    result = numpy.zeros((num), dtype="int32")
    for i in range(num):
        choiceTO = choicesTO[i]
        if choiceTO != 0 and is_position_empty(X_TO[i], choiceTO, data_format):
            if (not is_phase_2(X_TO[i], data_format)):
                result[i] = 1
            # phase 2
            else:
                adj = False
                # control if there is a friendly stone adjacent
                # find the adjacent ones
                adjacents = near[choiceTO]
                
                # verify if there is a player stone
                for j in range(1, 25):
                   
                    if adjacents[j] == 1 and is_position_player(X_TO[i], j, data_format):
                        adj = True
                        break
                if adj:
                    result[i] = 1
                else:
                    result[i] = 0
        else:
            result[i] = 0
    return result


# position: between 1 and 24
def is_position_player(X, position, data_format="binary raw"):
    if (data_format == "binary raw" or data_format == "binary rawer" or
        data_format == "binary rawest"):
        if position == 0:
            return False
        if X[position - 1] == 1:
            return True
        else:
            return False


# position: between 1 and 24
def is_position_enemy(X, position, data_format="binary raw"):
    if (data_format == "binary raw" or data_format == "binary rawer" or
        data_format == "binary rawest"):
        if position == 0:
            return False
        if X[position + 23] == 1:
            return True
        else:
            return False


# position: between 1 and 24
def is_position_empty(X, position, data_format="binary raw"):
    if (data_format == "binary raw" or data_format == "binary rawer" or
        data_format == "binary rawest"):
        if position == 0:
            return False
        if X[position + 47] == 1:
            return True
        else:
            return False


def is_phase_2(X, data_format="binary raw"):
    if (data_format == "binary raw"):
        if (X[109] != 0 and X[110] != 1):
            return True
        else:
            return False
    elif (data_format == "binary rawer"):
        if (X[91] != 0 and X[92] != 1):
            return True
        else:
            return False
    elif (data_format == "binary rawest"):
        # if there are no checkers in my hand
        if(X[72] == 0):
            n = 0
            # count the checkers on the board
            for i in range (0,24):
                if X[i] == 1:
                    n += 1
            # if there are more than 3 checkers, it's phase 2
            if n > 3:
                return True
            else:
                return False
        else:
            return False
        


def is_phase_1(X, data_format="binary raw"):
    if (data_format == "binary raw"):
        if (X[109] == 0):
            return True
        else:
            return False
    elif (data_format == "binary rawer"):
        if(X[91] == 0):
            return True
        else:
            return False
    elif (data_format == "binary rawest"):
        if(X[72] == 1):
            return True
        else:
            return False


def is_phase_3(X, data_format="binary raw"):
    if (data_format == "binary raw"):
        if (X[110] == 1):
            return True
        else:
            return False
    elif (data_format == "binary rawer"):
        if(X[92] == 1):
            return True
        else:
            return False
    elif (data_format == "binary rawest"):
        # if there are no checkers in my hand
        if(X[72] == 0):
            n = 0
            # count the checkers on the board
            for i in range (0,24):
                if X[i] == 1:
                    n += 1
            # if there are more than 3 checkers, it not phase 3
            if n > 3:
                return False
            else:
                return True
        else:
            return False


# in phase one must be 0
# in phase two there must be a player stone
def FROM_self_legality_test(choicesFROM, X_FROM, data_format="binary raw"):
    num = len(choicesFROM)
    result = numpy.zeros((num), dtype="int32")
    for i in range(num):
        choiceFROM = choicesFROM[i]
        # if phase one, check if the choice is no-position
        if is_phase_1(X_FROM[i], data_format):
            if choiceFROM == 0:
                result[i] = 1
            else:
                result[i] = 0
        # if not phase one, check if the position has a player stone
        else:
            if is_position_player(X_FROM[i], choiceFROM, data_format):
                result[i] = 1
            else:
                result[i] = 0
    return result


# 0 is legal
# must be a not milled enemy stone or every enemy stone must be in a mill
def REMOVE_self_legality_test(choicesREMOVE, X_REMOVE,
                              data_format="binary raw"):
    num = len(choicesREMOVE)
    result = numpy.zeros((num), dtype="int32")
    for i in range(num):
        choiceREMOVE = choicesREMOVE[i]
        if (choiceREMOVE == 0):
            result[i] = 1
        else:
            # verify if the position of the removal is occupied by an enemy
            if is_position_enemy(X_REMOVE[i], choiceREMOVE, data_format) :

                # verify if the removed one is aligned
                opta = row1[choiceREMOVE]
                optb = row2[choiceREMOVE]
                optc = column1[choiceREMOVE]
                optd = column2[choiceREMOVE]
                
                trisrow = 0
                triscol = 0
                
                # verify if in the column/row there are the enemy's stones
                trisrow = is_position_enemy(X_REMOVE[i], opta, data_format) and\
                          is_position_enemy(X_REMOVE[i], optb, data_format)
                triscol = is_position_enemy(X_REMOVE[i], optc, data_format) and\
                          is_position_enemy(X_REMOVE[i], optd, data_format)
                
                if (not trisrow) and (not triscol):
                    result[i] = 1
                else:
                    # verify if there is any not aligned stone
                    notaligned = False
                    for j in range (1, 25):
                        opta = row1[j]
                        optb = row2[j]
                        optc = column1[j]
                        optd = column2[j]
                        trisrow = 0
                        triscol = 0
                        # verify if in the column/row there are the enemy's stones
                        trisrow = is_position_enemy(X_REMOVE[i], opta, data_format) and\
                                  is_position_enemy(X_REMOVE[i], optb, data_format)
                        triscol = is_position_enemy(X_REMOVE[i], optc, data_format) and\
                                  is_position_enemy(X_REMOVE[i], optd, data_format)
                        if (not trisrow) and (not triscol):
                            notaligned = True
                    
                    if notaligned:
                        result[i] = 0
                    else:
                        result[i] = 1
                
            else:
                result[i] = 0
    return result


# returns a tuple of binary arrarys. Each arrays represent a legality test
# and each element of the array is 1 if the element has passed the test
# In order, the legality tests are:
# - TO_self, FROM_self, REMOVE_self
# - FROM-TO, REMOVE-FROM-TO
# - exceptREMOVE: TO_self, FROM_self and FROM-TO
# - WholeREMOVE: TO_self, FROM_self, REMOVE_self and REMOVE-FROM-TO
# - WholeMOVE: exceptREMOVE and WholeREMOVE
def get_legalities(TO_choice, FROM_choice, REMOVE_choice,
                     X_test, data_format="binary raw"):
    
    if TO_choice is not None:
        TO_self_leg = TO_self_legality_test(TO_choice, X_test, data_format)
    else:
        TO_self_leg = None

    if FROM_choice is not None:
        FROM_self_leg = FROM_self_legality_test(FROM_choice, X_test, data_format)
    else:
        FROM_self_leg = None

    if TO_choice is not None and FROM_choice is not None :
        FT_leg = FT_legality_test(TO_choice, FROM_choice, X_test, data_format)
    else:
        FT_leg = None

    if REMOVE_choice is not None:
        REMOVE_self_leg = REMOVE_self_legality_test(REMOVE_choice, X_test,
                                                    data_format)
    else:
        REMOVE_self_leg = None
    
    if (TO_choice is not None and FROM_choice is not None and
        REMOVE_choice is not None) :
        TFR_leg = TFR_legality_test(TO_choice, FROM_choice, REMOVE_choice,
                                          X_test, data_format)
    else:
        TFR_leg = None

    if TO_choice is not None and FROM_choice is not None :
        m1 = numpy.multiply(TO_self_leg, FROM_self_leg)
        exceptREMOVE = numpy.multiply(m1, FT_leg)
    else:
        exceptREMOVE = None
        m1 = None
        
    if (TO_choice is not None and FROM_choice is not None and
        REMOVE_choice is not None) :
        m2 = numpy.multiply(m1, REMOVE_self_leg)
        wholeREMOVE = numpy.multiply(m2, TFR_leg)
        wholeMOVE = numpy.multiply(wholeREMOVE, exceptREMOVE)
    else:
        wholeREMOVE = None
        wholeMOVE = None
        
    # TODO: improve this
    if (TO_choice is not None and REMOVE_choice is not None):
        exceptFROM = numpy.multiply(TO_self_leg, REMOVE_self_leg)
    else:
        exceptFROM = None        
        
    # TODO: improve this
    if (FROM_choice is not None and REMOVE_choice is not None):
        exceptTO = numpy.multiply(FROM_self_leg, REMOVE_self_leg)
    else:
        exceptTO = None

    return (TO_self_leg, FROM_self_leg, REMOVE_self_leg, FT_leg, TFR_leg,
            exceptREMOVE, wholeREMOVE, wholeMOVE, exceptFROM, exceptTO)

# if (109 == 0) fase 1
# if (110 == 1) fase 3
  