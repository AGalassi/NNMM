#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2016-2017, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"

import numpy
import theano
import lasagne
import sys
import socket


from dataprocessing import (process_state_binary, add_CHOICE_binary_raw,
                            process_game_line)
from legality import get_legalities
from networks import load_net

illegalTO = 0
illegalFROM = 0
illegalREMOVE = 0

whiteport = 5802
blackport = 5803


def main(netname='TEST-TFR', team = "white"):
    if (team == 'white'):
        play(netname, whiteport)
    elif (team == 'black'):
        play(netname, blackport)
    else:
        play(netname)
    sys.exit(0)

        

def init(name):

    inputTO, TOnetwork, numTO, dataformat = load_net(name + "_TO") 
    
    inputFROM, FROMnetwork, numFROM, dataformat = load_net(name + "_FROM") 

    inputREMOVE, REMOVEnetwork, numREMOVE, dataformat = load_net(name + "_REMOVE") 

    TOpred = lasagne.layers.get_output(TOnetwork, deterministic=True)
    TOpred_fn = theano.function(
        [inputTO], TOpred, name="TO prediction function")

    FROMpred = lasagne.layers.get_output(FROMnetwork, deterministic=True)
    FROMpred_fn = theano.function(
        [inputFROM], FROMpred, name="FROM prediction function")

    REMOVEpred = lasagne.layers.get_output(REMOVEnetwork, deterministic=True)
    REMOVEpred_fn = theano.function(
        [inputREMOVE], REMOVEpred, name="REMOVE prediction function")

    return ((TOpred_fn, numTO),
            (FROMpred_fn, numFROM),
            (REMOVEpred_fn, numREMOVE),
            dataformat)


def choose(TOnet, FROMnet, REMOVEnet, state, data_format):
        
    
    orderc = ['X', 'X', 'X']
    orderc[TOnet[1]] = 'T'
    orderc[FROMnet[1]] = 'F'
    orderc[REMOVEnet[1]] = 'R'
    order = "" + orderc[0] + orderc[1] + orderc[2]
    print("\tOrder: " + order)    
    
    nets = ['X', 'X', 'X']
    nets[TOnet[1]] = TOnet
    nets[FROMnet[1]] = FROMnet
    nets[REMOVEnet[1]] = REMOVEnet
    
    FIRSTpred_fn = nets[0][0]
    SECONDpred_fn = nets[1][0]
    THIRDpred_fn =nets[2][0]
    
    global illegalTO
    global illegalFROM
    global illegalREMOVE
    
    illegal = ['X', 'X', 'X']
    illegal[TOnet[1]] = illegalTO
    illegal[FROMnet[1]] = illegalFROM
    illegal[REMOVEnet[1]] = illegalREMOVE
    
    print("Choosing best move: ")
    print("\tchoosing FIRST position...")
    
    # TO choice
    states = [state]
    FIRSTstate = process_state_binary(states, data_format)
    
    FIRSTprob = FIRSTpred_fn(FIRSTstate)
    
    correct = False
    FIRSTchoice = 0
    while not correct:
        propchoice = numpy.argmax(FIRSTprob[0])
        if FIRSTprob[0][propchoice] >= 0:
            FIRSTchoice = [propchoice]
        else:
            print("No legal option aviable")
            sys.exit()
        print("\t\tTOchoice: " + str(FIRSTchoice[0]))
        
        vec_4_leg = [FIRSTchoice, None, None]
        
        legalities = get_legalities(vec_4_leg[TOnet[1]],
                                    vec_4_leg[FROMnet[1]],
                                    vec_4_leg[REMOVEnet[1]],
                                    FIRSTstate)
        
        if orderc[0] == 'T':
            legalities = legalities[0]
        elif orderc[0] == 'F':
            legalities = legalities[1]
        elif orderc[0] == 'R':
            legalities = legalities[2]
        
        if(legalities[0] == 0):
            FIRSTprob[0][FIRSTchoice[0]] = -1
            print("\t\t\tIllegal FIRST choice, retry")
            illegal[0] += 1
        else:
            correct = True

    print("\tchoosing SECOND position...")
    # SECOND choice
    SECONDstate = add_CHOICE_binary_raw(FIRSTstate, FIRSTchoice)
    SECONDprob = SECONDpred_fn(SECONDstate)
    correct = False
    SECONDchoice = 0
    while not correct:
        propchoice = numpy.argmax(SECONDprob[0])
        if SECONDprob[0][propchoice] >= 0:
            SECONDchoice = [propchoice]
        else:
            print("No legal option aviable")
            sys.exit()
        print("\t\tSECONDchoice: " + str(SECONDchoice[0]))
        
        vec_4_leg = [FIRSTchoice, SECONDchoice, None]
        
        legalities = get_legalities(vec_4_leg[TOnet[1]],
                                    vec_4_leg[FROMnet[1]],
                                    vec_4_leg[REMOVEnet[1]],
                                    SECONDstate)
        
        if orderc[2] == 'T':
            legal = legalities[9]
        elif orderc[2] == 'F':
            legal = legalities[8]
        elif orderc[2] == 'R':
            legal = legalities[5]
        
        if(legal[0] == 0):
            SECONDprob[0][SECONDchoice[0]] = -1
            print("\t\t\tIllegal FROM choice, retry")
            illegal[1] += 1
        else:
            correct = True
    
    print("\tchoosing THIRD position...")
    # REMOVE choice
    THIRDstate = add_CHOICE_binary_raw(SECONDstate, SECONDchoice)
    THIRDprob = THIRDpred_fn(THIRDstate)
    correct = False
    THIRDchoice = 0
    while not correct:
        propchoice = numpy.argmax(THIRDprob[0])
        if THIRDprob[0][propchoice] >= 0:
            THIRDchoice = [propchoice]
        else:
            print("No legal option aviable")
            sys.exit()
        print("\t\tTHIRDchoice: " + str(THIRDchoice[0]))
        
        vec_4_leg = [FIRSTchoice, SECONDchoice, THIRDchoice]
        
        legalities = get_legalities(vec_4_leg[TOnet[1]],
                                    vec_4_leg[FROMnet[1]],
                                    vec_4_leg[REMOVEnet[1]],
                                    SECONDstate)[7]
        
        if(legalities[0] == 0):
            THIRDprob[0][THIRDchoice[0]] = -1
            print("\t\t\tIllegal REMOVE choice, retry")
            illegal[2] += 1
        else:
            correct = True

    choices = [FIRSTchoice, SECONDchoice, THIRDchoice]

    return (choices[TOnet[1]][0],
            choices[FROMnet[1]][0],
            choices[REMOVEnet[1]][0])


# color: W for white, B for black
def play(name, port=whiteport):
    if (port == whiteport):
        print("Hi, player White")
    else:
        print("Hi, player Black")

    print("Loading networks")
    
    TOnet, FROMnet, REMOVEnet, data_format = init(name)
    print("\tNetworks loaded!")

    sys.stdout.flush()
    print("\tNetworks loaded")

    print("Connecting to the game on port " + str(port))
    sys.stdout.flush()
    socket, connection = connect(port)
    
    choices = 0
    
    global illegalTO
    global illegalFROM
    global illegalREMOVE
    print("\tConnected!")
    sys.stdout.flush()
    while(True):
        # listen to the socket
        
        print ("Waiting for message")
        sys.stdout.flush()

        data = receive(connection)
        datastr = str(data)
        print("String recieved: " + datastr)
        
        # processing the state
        state = process_game_line(datastr)
        print("State: " + str(state))
        sys.stdout.flush()
        # answering
        TOc, FROMc, REMOVEc = choose(TOnet, FROMnet, REMOVEnet,
                                     state, data_format)
        
        print("Choices: " + str(TOc) + " " + str(FROMc) + " " + str(REMOVEc))
        choices += 1
        meanIT = illegalTO * 1.0 / choices
        meanIF = illegalFROM * 1.0 / choices
        meanIR = illegalREMOVE * 1.0 / choices

        print("\tIllegal TO: " + str(illegalTO) +
              " (mean: " + str(meanIT) + ")")
        print("\tIllegal FROM: " + str(illegalFROM) +
              " (mean: " + str(meanIF) + ")")
        print("\tIllegal REMOVE: " + str(illegalREMOVE) +
              " (mean: " + str(meanIR) + ")")
    
        connection.send("" + str(TOc) + "\n")
        connection.send("" + str(FROMc) + "\n")
        connection.send("" + str(REMOVEc) + "\n")
        sys.stdout.flush()
    

def connect(port):
    # try to connect
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('', port))
    except socket.error as msg:
        print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
        exit(4)
    print("\tSocket binding completed")
    s.listen(100)
    conn, addr = s.accept()
    print("\tConnection established")

    return s, conn


def receive(s):
    MSGLEN = 30
    chunks = []
    bytes_recd = 0
    while bytes_recd < MSGLEN:
        chunk = s.recv(min(MSGLEN - bytes_recd, 4096))
        print(str(chunk))
        if chunk == '':
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd = bytes_recd + len(chunk)
    data = ''.join(chunks)

    return data


if __name__ == '__main__':
    kwargs = {}
    usage = ("Usage: %s netname team" % sys.argv[0])
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Neural Nine Men's Morris\n" +
              "Playing clinet:\n" +
              "Specify the name of the networks and the team " + 
              "color (because of the team port)")
    if len(sys.argv) == 3:
        kwargs['netname'] = sys.argv[1]
        kwargs['team'] = sys.argv[2]
        
        main(**kwargs)
    elif len(sys.argv) == 1 :
        main(**kwargs)
    else:
        print ("Wrong number of arguments: " + str(len(sys.argv))+"\n" + usage)



    