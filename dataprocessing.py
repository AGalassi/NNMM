#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2016-2017, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"

import numpy


class State:

    # the 24 position, number of mine/enemy checkers on hand and on board
    def __init__(self, positions, moh, eoh, mob, eob):
        self.positions = positions
        self.moh = int(moh)
        self.eoh = int(eoh)
        self.mob = int(mob)
        self.eob = int(eob)
        self.my_phase = 2
        self.enemy_phase = 2
        if self.moh > 0:
            self.my_phase = 1
        elif (self.mob == 3):
            self.my_phase = 3
        elif (self.mob < 3):
            self.my_phase = 4

        if (self.eoh > 0):
            self.enemy_phase = 1
        elif (self.eob == 3):
            self.enemy_phase = 3
        elif (self.eob < 3):
            self.enemy_phase = 4

    def __str__(self):
        stringa = ""
        for pos in self.positions:
            stringa += " " + pos
            # print(stringa)
        stringa += ", MoH: " + str(self.moh)
        # print(stringa)
        stringa += ", EoH: " + str(self.eoh)
        # print(stringa)
        stringa += ", MoB: " + str(self.mob)
        # print(stringa)
        stringa += ", EoB: " + str(self.eob)
        # print(stringa)
        stringa += ", MP: " + str(self.my_phase)
        # print(stringa)
        stringa += ", EP: " + str(self.enemy_phase)

        return stringa

    def __eq__(self, other):
        for i in range(0, 24):
            if(self.positions[i] != other.positions[i]):
                return False
        if (self.moh != other.moh or self.eoh !=
                other.eoh or self.mob != other.mob or self.eob != other.eob):
            return False
        return True

    def __ne__(self, other):
        for i in range(0, 24):
            if(self.positions[i] != other.positions[i]):
                return True
        if (self.moh != other.moh or self.eoh !=
                other.eoh or self.mob != other.mob or self.eob != other.eob):
            return True
        return False
    
    def __hash__(self):
        return hash(self.positions, self.moh, self.eoh, self.mob, self.eob)
    
    def to_dataset_string(self):
        stringa = ""
        for pos in self.positions:
            stringa +=  pos
            # print(stringa)
            
        stringa += str(self.moh)
        stringa += str(self.eoh)
        stringa += str(self.mob)
        stringa += str(self.eob)
        return stringa
    
    def to_board(self):
        stringa = ""
        pos = self.positions
        
        stringa += (pos[0]+"--"+pos[1]+"--"+pos[2]+"\n"+
                   "|"+ pos[3]+"-"+pos[4]+"-"+pos[5]+"|\n"+
                   "||"+pos[6]+pos[7]+pos[8]+"||\n"+
                   pos[9]+pos[10]+pos[11]+" "+pos[12]+pos[13]+pos[14]+"\n"+
                   "||"+pos[15]+pos[16]+pos[17]+"||\n"+
                   "|"+ pos[18]+"-"+pos[19]+"-"+pos[20]+"|\n"+
                   pos[21]+"--"+pos[22]+"--"+pos[23]+"\n")
            
        stringa += "MoH: " + str(self.moh)
        # print(stringa)
        stringa += ", EoH: " + str(self.eoh)
        # print(stringa)
        stringa += ", MoB: " + str(self.mob)
        # print(stringa)
        stringa += ", EoB: " + str(self.eob)
        # print(stringa)
        stringa += ", MP: " + str(self.my_phase)
        # print(stringa)
        stringa += ", EP: " + str(self.enemy_phase)
        return stringa


def convert_move(mossa, fase):
    """
    Decompose a move string into numbers
    
    Given a dataset move string, it is decomposed into its 3 parts as numbers
    which represent the board positions

    Parameters
    ----------
    mossa : string
        The move string
    fase : int
        the phase number
    
    Returns
    -------
    int
        The TO position
    int
        Whether there is a FROM position
    int
        The FROM position
    int
        Whether there is a REMOVE position
    int
        The REMOVE position

    """
    conversione = {"a7": 1, "d7": 2, "g7": 3, "b6": 4, "d6": 5, "f6": 6,
                   "c5": 7,
                   "d5": 8,
                   "e5": 9,
                   "a4": 10,
                   "b4": 11,
                   "c4": 12,
                   "e4": 13,
                   "f4": 14,
                   "g4": 15,
                   "c3": 16,
                   "d3": 17,
                   "e3": 18,
                   "b2": 19,
                   "d2": 20,
                   "f2": 21,
                   "a1": 22,
                   "d1": 23,
                   "g1": 24}
    
    has_REMOVE = 0
    has_FROM = 0
    FROM = 0
    REMOVE = 0
    TO = 0
    if(fase == 2 or fase == 3):
        m1 = mossa[2:4]
        TO = conversione[m1]
        m2 = mossa[0:2]
        FROM = conversione[m2]
        has_FROM = 1
        if (len(mossa) > 5):
            m3 = mossa[4:6]
            REMOVE = conversione[m3]
            has_REMOVE = 1
    elif(fase == 1):
        m1 = mossa[0:2]
        TO = conversione[m1]
        if (len(mossa) > 3):
            m3 = mossa[2:4]
            REMOVE = conversione[m3]
            has_REMOVE = 1
    return TO, has_FROM, FROM, has_REMOVE, REMOVE

def reconvert_move(move):
    """
    Converts a move tuple into a string
    
    Given a move tuple, constructs a database string which represents the move.
    The move is given by one up to three coordinates (TO-FROM-REMOVE)

    Parameters
    ----------
    move: int[]
        A move tuple, as outputed as the convert_move function
    
    Returns
    -------
    string
        The string move in the same format of the dataset

    """
    conversion = {0: "", 1: "a7", 2: "d7", 3: "g7", 4: "b6", 5: "d6", 6: "f6",
                   7: "c5",
                   8: "d5",
                   9: "e5",
                   10: "a4",
                   11: "b4",
                   12: "c4",
                   13: "e4",
                   14: "f4",
                   15: "g4",
                   16: "c3",
                   17: "d3",
                   18: "e3",
                   19: "b2",
                   20: "d2",
                   21: "f2",
                   22: "a1",
                   23: "d1",
                   24: "g1"}
    p = conversion[move[0]]
    f = conversion[move[2]]
    r = conversion[move[4]]
    return p+f+r
 

# warp the state and the move applying inversed warping
def warp(state, move, warp_v):
    """
    Warps the state and the move applying inversed warping
    
    

    Parameters
    ----------
    state : State
        The game state
    move : int[]
        A move tuple
    warp_v : int[]
        The warping array
    
    Returns
    -------
    State
        The warped state
    int[]
        The warped move tuple

    """
    pos = state.positions
    new_pos = ""

    j1 = 0
    j2 = 0
    j3 = 0

    # for each new position I look at what was the previous one and what was in
    # there for the moves, when I found the original position,
    # I write the warped one
    for i in range(0, 24):
        n = warp_v[i]
        new_pos += pos[n - 1]

        if (move is not None):
            if(move[0] == n):
                j1 = i + 1
            if(move[2] == n):
                j2 = i + 1
            if(move[4] == n):
                j3 = i + 1
    
    
    if (move is not None):    
        new_move = j1, move[1], j2, move[3], j3
    else:
        new_move = None

    new_state = State(new_pos, state.moh, state.eoh, state.mob, state.eob)

    return new_state, new_move



def add_symmetries(stato, move, states, moves):
    """
    Creates the symmetries of a couple state-move and add them to the lists
    
    Apply all the possible symmetries to the couple state-move given and
    add the warped couples to the states-moves lists.
    If the same state is obtained by different warps of the same original
    state, only one is added.
    

    Parameters
    ----------
    stato : State
        The game state
    move : int[]
        A move tuple
    states : States[]
        The list of states
    moves: int[][]
        The list of moves
    
    Returns
    -------
    """
    rotate = (22, 10, 1, 19, 11, 4, 16, 12, 7, 23, 20, 17,
              8, 5, 2, 18, 13, 9, 21, 14, 6, 24, 15, 3)
    mirrorhor = (3, 2, 1, 6, 5, 4, 9, 8, 7, 15, 14, 13, 12,
                 11, 10, 18, 17, 16, 21, 20, 19, 24, 23, 22)
    insideout = (7, 8, 9, 4, 5, 6, 1, 2, 3, 12, 11, 10, 15,
                 14, 13, 22, 23, 24, 19, 20, 21, 16, 17, 18)
    middleout = (4, 5, 6, 1, 2, 3, 7, 8, 9, 11, 10, 12, 13,
                 15, 14, 16, 17, 18, 22, 23, 24, 19, 20, 21)

    rot1_s, rot1_m = warp(stato, move, rotate)
    rot2_s, rot2_m = warp(rot1_s, rot1_m, rotate)
    rot3_s, rot3_m = warp(rot2_s, rot2_m, rotate)

    mir_s, mir_m = warp(stato, move, mirrorhor)
    rot1mir_s, rot1mir_m = warp(rot1_s, rot1_m, mirrorhor)
    rot2mir_s, rot2mir_m = warp(rot2_s, rot2_m, mirrorhor)
    rot3mir_s, rot3mir_m = warp(rot3_s, rot3_m, mirrorhor)

    ino_s, ino_m = warp(stato, move, insideout)

    rot1ino_s, rot1ino_m = warp(rot1_s, rot1_m, insideout)
    rot2ino_s, rot2ino_m = warp(rot2_s, rot2_m, insideout)
    rot3ino_s, rot3ino_m = warp(rot3_s, rot3_m, insideout)

    mirino_s, mirino_m = warp(mir_s, mir_m, insideout)
    rot1mirino_s, rot1mirino_m = warp(rot1mir_s, rot1mir_m, insideout)
    rot2mirino_s, rot2mirino_m = warp(rot2mir_s, rot2mir_m, insideout)
    rot3mirino_s, rot3mirino_m = warp(rot3mir_s, rot3mir_m, insideout)

    tempstates = [rot1_s, rot2_s, rot3_s,
                  mir_s, rot1mir_s, rot2mir_s, rot3mir_s,
                  ino_s, rot1ino_s, rot2ino_s, rot3ino_s,
                  mirino_s, rot1mirino_s, rot2mirino_s, rot3mirino_s]

    tempmoves = [rot1_m, rot2_m, rot3_m,
                 mir_m, rot1mir_m, rot2mir_m, rot3mir_m,
                 ino_m, rot1ino_m, rot2ino_m, rot3ino_m,
                 mirino_m, rot1mirino_m, rot2mirino_m, rot3mirino_m]

    templen = len(tempstates)
    # if there are only three checkers for both player in the board, there
    # is another symmetry to consider
    if stato.mob == 3 and stato.eob == 3:
        for i in range(0, templen):
            tempstate = tempstates[i]
            tempmove = tempmoves[i]
            mo_s, mo_m = warp(tempstate, tempmove, middleout)
            tempstates.append(mo_s)
            tempmoves.append(mo_m)
        templen = len(tempstates)

    # calculate the index of the last element (the state)
    s_len = len(states) - 1

    # for every possible state
    for i in range(0, templen):
        tempstate = tempstates[i]
        tempmove = tempmoves[i]
        duplicate = False
        # verify is not a copy of another symmetry
        for finalstate in states[s_len:]:
            if finalstate == tempstate:
                duplicate = True
                break
        if not duplicate:
            states.append(tempstate)
            if (moves is not None):
                moves.append(tempmove)


# carica il dataset da un file
def load_dataset(filename):
    """
    Loads a dataset of couples move-state from a file, expanding the symmetries
    
    

    Parameters
    ----------
    filename : string
        The name of the dataset file
    
    Returns
    -------
    States[]
        The list of states
    int[][]
        The list of moves
    """

    # carico il dataset
    dataset_file = open(filename, 'r')
    dataset_list = dataset_file.read().splitlines()

    states = []
    moves = []

    # ogni linea del file è una coppia stato-mossa
    for line in dataset_list:
        state, move = process_dataset_line(line)

        states.append(state)
        moves.append(move)

        # carico gli stati e le mosse simmetriche
        add_symmetries(state, move, states, moves)

    dataset_file.close()

    return states, moves


def load_expanded_dataset(filename):
    """
    Loads a dataset of couples move-state from a file,
    without expanding the symmetries
    
    

    Parameters
    ----------
    filename : string
        The name of the dataset file
    
    Returns
    -------
    States[]
        The list of states
    int[][]
        The list of moves
    """

    # carico il dataset
    dataset_file = open(filename, 'r')
    dataset_list = dataset_file.read().splitlines()

    states = []
    moves = []

    # ogni linea del file è una coppia stato-mossa
    for line in dataset_list:
        state, move = process_dataset_line(line)

        states.append(state)
        moves.append(move)

    dataset_file.close()

    return states, moves


def load_states_dataset(filename):
    """
    Loads a dataset of states from a file, expanding the symmetries
    
    

    Parameters
    ----------
    filename : string
        The name of the dataset file
    
    Returns
    -------
    States[]
        The list of states
    None
    """
    # carico il dataset
    dataset_file = open(filename, 'r')
    dataset_list = dataset_file.read().splitlines()

    states = []
    moves = []

    # ogni linea del file è una coppia stato-mossa
    for line in dataset_list:
        state = process_states_dataset_line(line)

        states.append(state)
        move = 1, 0, 0, 0, 0
        moves.append(move)

        # carico gli stati e le mosse simmetriche
        add_symmetries(state, move, states, moves)

    dataset_file.close()

    return states, moves


def load_expanded_states_dataset(filename):
    """
    Loads a dataset of states from a file, without expanding the symmetries
    
    

    Parameters
    ----------
    filename : string
        The name of the dataset file
    
    Returns
    -------
    States[]
        The list of states
    None
    """

    # carico il dataset
    dataset_file = open(filename, 'r')
    dataset_list = dataset_file.read().splitlines()

    states = []
    moves = []

    # ogni linea del file è una coppia stato-mossa
    for line in dataset_list:
        state = process_states_dataset_line(line)

        states.append(state)
        move = 0, 0, 0, 0, 0
        moves.append(move)

    dataset_file.close()

    return states, moves
    

def expand_dataset(filename, onlystates = False):
    """
    Loads a dataset of states from a file and writes the expanded version of
    the dataset.
    
    

    Parameters
    ----------
    filename : string
        The name of the dataset file
    onlystates : boolean
        Whether the dataset is made only by states or, on the contrary, by 
        moves too
    
    Returns
    -------
    """
    if (onlystates):
        A, B = load_states_dataset(filename)
    else:
        A, B = load_dataset(filename)
    print("Dataset loaded")
    efile = open ("EXPANDED_DATASET.txt", "w")
    for i in range(len(A)):
        state = A[i]
        for pos in state.positions:
            efile.write(str(pos))
        efile.write(str(state.moh))
        efile.write(str(state.eoh))
        efile.write(str(state.mob))
        efile.write(str(state.eob))
        if (not onlystates):
            efile.write("-")
            move = B[i]
            efile.write(reconvert_move(move))
        efile.write("\n")
    efile.close()


def process_dataset_line(line):
    position = [None] * 24

    for j in range(0, 24):
        position[j] = line[j]

    # carico lo stato
    state = State(position, line[24], line[25], line[26], line[27])

    # carico la mossa
    move = convert_move(line[29:], state.my_phase)

    return state, move


def process_states_dataset_line(line):
    position = [None] * 24

    for j in range(0, 24):
        position[j] = line[j]

    # carico lo stato
    state = State(position, line[24], line[25], line[26], line[27])

    return state


def process_game_line(line):
    position = [None] * 24

    while(line[0]!='O' and line[0]!='E' and line[0]!='M'):
        print(line)
        line=line[1:]
    
    for j in range(0, 24):
        position[j] = line[j]

    # carico lo stato
    state = State(position, line[24], line[25], line[26], line[27])

    return state


def process_move_onlyTO(moves):
    length = len(moves)

    pm = numpy.zeros((length), dtype="uint8")

    i = 0

    for tupla in moves:
        pm[i] = tupla[0]

        i = i + 1

    return pm


def process_move_onlyFROM(moves):
    length = len(moves)

    pm = numpy.zeros((length), dtype="uint8")

    i = 0

    for tupla in moves:
        pm[i] = tupla[2]

        i = i + 1

    return pm


def process_move_onlyREMOVE(moves):
    length = len(moves)

    pm = numpy.zeros((length), dtype="uint8")

    i = 0

    for tupla in moves:
        pm[i] = tupla[4]

        i = i + 1

    return pm


# preprocessing della mossa per averla nel formato desiderato
def process_move(moves):
    length = len(moves)

    pm = numpy.zeros((length, 5), dtype="uint8")

    i = 0

    for tupla in moves:
        pm[i][0] = tupla[0]
        pm[i][1] = tupla[1]
        pm[i][2] = tupla[2]
        pm[i][3] = tupla[3]
        pm[i][4] = tupla[4]

        i = i + 1
    
    return pm



def add_TO_binary(states, TO_choices):
    return add_CHOICE_binary_raw(states, TO_choices)


def add_FROM_binary(states, FROM_choices):
    return add_CHOICE_binary_raw(states, FROM_choices)


def add_REMOVE_binary(states, REMOVE_choices):
    return add_CHOICE_binary_raw(states, REMOVE_choices)


def add_CHOICE_binary_raw(states, choices):
    
    length = len(states)
    if (length <= 0):
        return numpy.zeros((0, 0), dtype="uint8")

    slen = len(states[0])

    ps = numpy.zeros((length, slen + 25), dtype="uint8")

    for i in range(length):
        for j in range(0, slen):
            ps[i][j] = states[i][j]

        for j in range(0, 25):
            if choices[i] == j:
                ps[i][j+slen] = 1
            else:
                ps[i][j+slen] = 0
    return ps


# preprocessing dello stato per averlo nel formato desiderato
def process_state_binary(states, data_format):
    # Process the list of States object into a matrix in which each row is a
    # vector representation of the state

    length = len(states)

    # print(length)

    ps = numpy.zeros((length, 114), dtype="uint8")
    
    if (data_format == "binary rawer"):
        ps = numpy.zeros((length, 96), dtype="uint8")
    elif (data_format == "binary rawest"):
        ps = numpy.zeros((length, 90), dtype="uint8")
        

    i = 0
    for state in states:

        # print(state)

        # mine checkers
        for j in range(0, 24):
            if(state.positions[j] == 'M'):
                ps[i][j] = 1
            else:
                ps[i][j] = 0

        # enemy checkers
        for j in range(24, 48):
            if(state.positions[j - 24] == 'E'):
                ps[i][j] = 1
            else:
                ps[i][j] = 0

        # empty positions
        for j in range(48, 72):
            if(state.positions[j - 48] == 'O'):
                ps[i][j] = 1
            else:
                ps[i][j] = 0

        # number of checker in hands as 9 digit binary number.
        # the number of ones indicates the number of checker
        for j in range(72, 72 + state.moh):
            ps[i][j] = 1
        for j in range(72 + state.moh, 81):
            ps[i][j] = 0
        for j in range(81, 81 + state.eoh):
            ps[i][j] = 1
        for j in range(81 + state.eoh, 90):
            ps[i][j] = 0

        if (data_format == "binary rawer"):
            # game phase
            for j in range(90, 90 + state.my_phase):
                ps[i][j] = 1
            for j in range(90 + state.my_phase, 93):
                ps[i][j] = 0
            for j in range(93, 93 + state.enemy_phase):
                ps[i][j] = 1
            for j in range(93 + state.enemy_phase, 96):
                ps[i][j] = 0
        
        elif(data_format == "binary raw"):
             # checkers on board
            for j in range(90, 90 + state.mob):
                ps[i][j] = 1
            for j in range(90 + state.mob, 99):
                ps[i][j] = 0
            for j in range(99, 99 + state.eob):
                ps[i][j] = 1
            for j in range(99 + state.eob, 108):
                ps[i][j] = 0
    
            # game phase
            for j in range(108, 108 + state.my_phase):
                ps[i][j] = 1
            for j in range(108 + state.my_phase, 111):
                ps[i][j] = 0
            for j in range(111, 111 + state.enemy_phase):
                ps[i][j] = 1
            for j in range(111 + state.enemy_phase, 114):
                ps[i][j] = 0

        i += 1

    return ps


# preprocessing dello stato per averlo nel formato desiderato
def process_state_notbinary(states):
    # Process the list of States object into a matrix in which each row is a
    # vector representation of the state

    length = len(states)

    # print(length)

    ps = numpy.zeros((length, 30), dtype="uint8")

    i = 0
    for state in states:

        # print(state)

        # mine checkers
        for j in range(0, 24):
            if(state.positions[j] == 'M'):
                ps[i][j] = 1
            elif(state.positions[j] == 'E'):
                ps[i][j] = -1
            else:
                ps[i][j] = 0

        ps[i][24] = state.moh
        ps[i][25] = state.eoh
        ps[i][26] = state.mob
        ps[i][27] = state.eob
        ps[i][28] = state.my_phase
        ps[i][29] = state.enemy_phase

        i += 1

    return ps



# preprocessing dello stato per averlo nel formato desiderato. DA SCRIVERE
def process_state_augmented(states):
    # Process the list of States object into a matrix in which each row is a
    # vector representation of the state

    trisses = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15),
               (16, 17, 18), (19, 20, 21), (22, 23, 24), (1, 10, 12),
               (4, 11, 19), (7, 12, 16), (2, 5, 8), (17, 20, 23), (9, 13, 18),
               (6, 14, 21), (3, 15, 24))

    length = len(states)

    # print(length)

    ps = numpy.zeros((length, 96), dtype="uint8")

    i = 0
    for state in states:

        # print(state)

        # as raw processing
        for j in range(0, 24):
            if(state.positions[j] == 'M'):
                ps[i][j] = 1
            else:
                ps[i][j] = 0

        for j in range(24, 48):
            if(state.positions[j - 24] == 'E'):
                ps[i][j] = 1
            else:
                ps[i][j] = 0
        for j in range(48, 72):
            if(state.positions[j - 48] == 'O'):
                ps[i][j] = 1
            else:
                ps[i][j] = 0

        for j in range(72, 72 + state.moh):
            ps[i][j] = 1
        for j in range(72 + state.moh, 81):
            ps[i][j] = 0
        for j in range(81, 81 + state.eoh):
            ps[i][j] = 1
        for j in range(81 + state.eoh, 90):
            ps[i][j] = 0

       
        # game phase
        for j in range(90, 90 + state.my_phase):
            ps[i][j] = 1
        for j in range(90 + state.my_phase, 93):
            ps[i][j] = 0
        for j in range(93, 93 + state.enemy_phase):
            ps[i][j] = 1
        for j in range(93 + state.enemy_phase, 96):
            ps[i][j] = 0

        # for each tris indicates if it's empty, full with white checkers
        # or full with black checkers
        for j in range(118, 118+16):
            tris = trisses[j-118]
            pos1 = tris[0]-1
            pos2 = tris[1]-1
            pos3 = tris[2]-1

            if(state.positions[pos1] == 'W' and
               state.positions[pos2] == 'W' and
               state.positions[pos3] == 'W'):
                ps[i][j] = 1
            else:
                ps[i][j] = 0

            if(state.positions[pos1] == 'B' and
               state.positions[pos2] == 'B' and
               state.positions[pos3] == 'B'):
                ps[i][j+16] = 1
            else:
                ps[i][j+16] = 0

            if(state.positions[pos1] == 'E' and
               state.positions[pos2] == 'E' and
               state.positions[pos3] == 'E'):
                ps[i][j+16+16] = 1
            else:
                ps[i][j+16+16] = 0

        # for each position verify if it closes a white or black mill
        for j in range(150, 174):
            ps[i][j] = 0
            pos = state.position[j-150]
            # if the position is empty, I check the nearby positions
            if pos == 'E':

                # I found in which trisses is pos involved
                for tris in trisses:
                    tocheck = []
                    for position in tris:
                        if position != pos+1:
                            tocheck.append(position-1)

                    if (len(tocheck) == 2 and
                            state.positions[tocheck[0]] == 'W' and
                            state.positions[tocheck[1]] == 'W'):
                        ps[i][j] = 1

        for j in range(174, 198):
            ps[i][j] = 0
            pos = state.position[j-174]
            # if the position is empty, I check the nearby positions
            if pos == 'E':

                # I found in which trisses is pos involved
                for tris in trisses:
                    tocheck = []
                    for position in tris:
                        if position != pos+1:
                            tocheck.append(position-1)

                    if (len(tocheck) == 2 and
                            state.positions[tocheck[0]] == 'B' and
                            state.positions[tocheck[1]] == 'B'):
                        ps[i][j] = 1

        i += 1

    return ps
    
    
    
def load_indexes(filename):
    index_file = open(filename, 'r')
    dataset_list = index_file.read().splitlines()
    list = []
    for line in dataset_list:
        line.rstrip('\n')
        list.append(int(line))
    return list

    
    
def write_indexes(filename, indexes):
    index_file = open(filename, 'w')
    for index in indexes:
        index_file.write(str(index) + "\n")
    
    
