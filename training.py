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
import lasagne.regularization as rgl
import sys
import random

from networks import (build_ffnet, build_densenet, build_resnet,
                      load_net_weights, save_net_weights)
from dataprocessing import (load_dataset, load_expanded_dataset,
                            process_state_binary, process_move_onlyTO,
                            process_move_onlyFROM, process_move_onlyREMOVE,
                            add_CHOICE_binary_raw)


def train(name='prova',
          datasetname="15vsALL.resultall.map.txt", expanded=False,
             vset_size=0.05, tset_size=0,
             testsetname="",
             movepart= "TO",
             order = "TFR",
             batch_size=2000, num_epochs=100,
             patience=10,
             nettype=2,
             neurons=[200, 300],
             blocks=10,
             lr_alfa0=0.001, b1=0.99, b2=0.999,
             lr_annealing=True,
             lr_k=0.1,
             dropi=0, drop=0,
             regularization=True,
             reg_type=rgl.l1,
             reg_weight=0.001,
             normalization=False,
             load=False, initial_epoch=0,
             data_format="binary raw"):
    
    if(vset_size < 0 or tset_size < 0):
        sys.exit(0)
    
    print("Loading data...")
    # Load the dataset and creates the symmetries
    if expanded:
        A, B = load_expanded_dataset(datasetname)
    else:
        A, B = load_dataset(datasetname)
    
    print("Dataset loaded: " + str(len(B)) + " data")

    # if the size of the set are expressed as decimal, they are the percentage
    # of the total set
    if(vset_size < 1):
        vset_size = vset_size * len(A)
    if(tset_size < 1):
        tset_size = tset_size * len(A)

    vset_size = int(vset_size)
    tset_size = int(tset_size)

    indexes = range(len(A))
    
    random.shuffle(indexes)
    
    A_train = []
    A_val = []
    A_test = []
    B_train = []
    B_val = []
    B_test = []
    
    val_ind = []
    test_ind = []

    if(testsetname != "" and vset_size > 0):
    # load a different test set from a file
        A_test, B_test = load_dataset(testsetname)
        train_ind, val_ind = indexes[:-(vset_size)], indexes[-(vset_size):]
        
    elif(tset_size > 0 and vset_size > 0):  # different validation and test set
        train_ind = indexes[:-(vset_size + tset_size)]
        val_ind = indexes[-(vset_size + tset_size):-tset_size]
        test_ind = indexes[-tset_size:]
        
    elif(vset_size > 0):  # no test set, only validation and train set
        train_ind = indexes[:-(vset_size + tset_size)]
        val_ind = indexes[-(vset_size):]
        
    else:  # only one set, used both for training and validation
        A_train, A_val = A, A
        B_train, B_val = B, B


    for index in train_ind:
        A_train.append(A[index])       
        B_train.append(B[index])
        
    for index in val_ind:
        A_val.append(A[index])       
        B_val.append(B[index])
        
    for index in test_ind:
        A_test.append(A[index])       
        B_test.append(B[index])

    print("Process data...")

    X_train = process_state_binary(A_train, data_format)
    X_val = process_state_binary(A_val, data_format)
    X_test = process_state_binary(A_test, data_format)
    
    TO_train = process_move_onlyTO(B_train)
    TO_val = process_move_onlyTO(B_val)
    TO_test = process_move_onlyTO(B_test)
    
    FROM_train = process_move_onlyFROM(B_train)
    FROM_val = process_move_onlyFROM(B_val)
    FROM_test = process_move_onlyFROM(B_test)
    
    REMOVE_train = process_move_onlyREMOVE(B_train)
    REMOVE_val = process_move_onlyREMOVE(B_val)
    REMOVE_test = process_move_onlyREMOVE(B_test)

    for char in order:
        if (char == 'T'):
            if (movepart == "TO"):
                y_train = process_move_onlyTO(B_train)
                y_val = process_move_onlyTO(B_val)
                y_test = process_move_onlyTO(B_test)
                break
            else:
                X_train = add_CHOICE_binary_raw(X_train, TO_train)
                X_val = add_CHOICE_binary_raw(X_val, TO_val)
                X_test = add_CHOICE_binary_raw(X_test, TO_test)
        
        elif (char == 'F'):
            if (movepart == "FROM"):
                y_train = process_move_onlyFROM(B_train)
                y_val = process_move_onlyFROM(B_val)
                y_test = process_move_onlyFROM(B_test)
                break
            else:
                X_train = add_CHOICE_binary_raw(X_train, FROM_train)
                X_val = add_CHOICE_binary_raw(X_val, FROM_val)
                X_test = add_CHOICE_binary_raw(X_test, FROM_test)
        
        elif (char == 'R'):
            if (movepart == "REMOVE"):
                y_train = process_move_onlyREMOVE(B_train)
                y_val = process_move_onlyREMOVE(B_val)
                y_test = process_move_onlyREMOVE(B_test)
                break
            else:
                X_train = add_CHOICE_binary_raw(X_train, REMOVE_train)
                X_val = add_CHOICE_binary_raw(X_val, REMOVE_val)
                X_test = add_CHOICE_binary_raw(X_test, REMOVE_test)

    print ("Data processed!\n")

    print("Using " + str(len(X_train)) + " data for training, " +
          str(len(X_val)) + " for validation and " +
          str(len(X_test)) + " for testing \n")
    
    if load:
        results_file = open(name + "_" + movepart + ".txt", 'a')
    else:
        results_file = open(name + "_" + movepart + ".txt", 'w')
    results_file.write("TRAINED ON: " + datasetname +
                       "\tVal: " + str(vset_size) +
                       "\tTest: " + str(tset_size) + "\n")
    results_file.close()

    time1 = time.time()
    do_training(X_train, X_val, X_test, y_train, y_val, y_test,
                name=name, movepart=movepart, order=order,
                batch_size=batch_size, num_epochs=num_epochs,
                patience=patience,
                nettype=nettype,
                neurons=neurons,
                blocks=blocks,
                lr_alfa0=lr_alfa0, b1=b1, b2=b2,
                lr_annealing=lr_annealing, lr_k=lr_k,
                dropi=dropi, drop=drop,
                regularization=regularization,
                reg_type=reg_type, reg_weight=reg_weight,
                normalization=normalization,
                load=load, initial_epoch=initial_epoch,
                data_format = data_format)
    time2 = time.time()
    totaltime = time2-time1
    print("\tTIME OCCURRED: " + str(totaltime))


# lr: learning rate
# b1 and b2: parameters of the adam update function
# num_epochs: number of training epochs

def do_training(X_train, X_val, X_test, y_train, y_val, y_test,
                movepart,
                order,
                nettype,
                name, batch_size,
                num_epochs, patience,
                lr_alfa0, b1, b2,
                neurons=[200, 200, 100, 50],
                dropi=0, drop=0, blocks=3,
                lr_annealing=False,
                lr_k=0.03,
                masked=False,
                regularization=False,
                reg_type=rgl.l1,
                reg_weight=0.01,
                legality_penalty=False, legality_weight=1,
                normalization=False,
                load=False, initial_epoch=0,
                data_format = "binary raw"):

    print("Starting training " + str(name))
    input_size = len(X_train[0])

    lr = theano.shared(lr_alfa0)
    # Prepare Theano variables for inputs and targets
    # 2 Dimensions: # of state and the bits of the state
    input_var = T.imatrix('inputs')
    # 1 dimension: the answer
    target_var = T.ivector('targets')

    # mask_var = None
    # if masked:
    #    mask_var = T.imatrix('masks')

    print ("Building network...")

    config_string = "trainset: " + str(len(y_train)) + "\tbatch_size: " +\
        str(batch_size) + "\tvalset: " +\
        str(len(y_val)) + "\ttestset: " + str(len(y_test)) + "\n"
        
    if(movepart == "FROM" or movepart == "REMOVE"):
        nout = 25
    elif (movepart == "TO"):
        nout = 25
        # nout = 24
    else:
        print("Unknown movepart: " + movepart)
        sys.exit()

    
    # Create neural network
    if (nettype == 1):
        network = build_ffnet(input_var=input_var,
                              input_size=input_size,
                              neurons=neurons,
                              pi=dropi, p=drop,
                              norm=normalization,
                              nout=nout)
        config_string += "ffnet\t" + str(neurons)
    elif (nettype == 2):
        network = build_resnet(input_var=input_var,
                               input_size=input_size,
                               neurons=neurons,
                               blocks=blocks,
                               pi=dropi, p=drop,
                               norm=normalization,
                               nout=nout)
        config_string += "resnet\t" + str(neurons[0]) + ", " +\
                          str(blocks) + " x " + str(neurons[1]) + " (x2)"

    elif (nettype == 3):
        network = build_densenet(input_var,
                                 input_size,
                                 neurons, blocks,
                                 dropi, drop,
                                 norm=normalization,
                                 nout=nout)
        
        config_string += "densenet\t" + str(neurons[0]) + ", " +\
                          str(blocks) + " x " + str(neurons[1])

    config_string += "\ninput\t" + str(input_size) + "\t" + data_format +\
                     "\t" + order + "\noutput\t" + movepart + "\t" +\
                     str(nout) + "\ndropout\t" + str(dropi) + "\t" + str(drop)
    if masked:
        config_string += "\nmasked training"

    config_string += "\nbatch normalization\t"
    if normalization:
        config_string += "yes"
    else:
        config_string += "no"

    if(load):
        network = load_net_weights(network, name + '_' + movepart)

    print ("BUILT!\nBuilding functions...")

    # Network output
    prediction = lasagne.layers.get_output(network)
    pred_fn = theano.function(
            [input_var], prediction, name="prediction function")
    
    # Network output, deterministic (skip the dropout)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    
    test_pred_fn = theano.function(
        [input_var], prediction, name="test prediction function")
    choice = T.argmax(prediction, axis=1)
    test_choice = T.argmax(test_prediction, axis=1)

    # loss function
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

    loss_fn = theano.function(
        [input_var, target_var], loss, name="loss function")

    loss = loss.mean()

    if regularization:
        config_string += "\nregularization\t"
        config_string += str(reg_type.func_name) + "\t"
        config_string += str(reg_weight)
        reg = rgl.regularize_layer_params(network, reg_type) * reg_weight
        loss = loss + reg

    
    # Update function
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params,
                                   learning_rate=lr, beta1=b1, beta2=b2)

    # deterministic loss function
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    if(regularization):
        test_loss = test_loss + reg

    
    # accuracy function
    test_acc = T.mean(T.eq(test_choice, target_var),
                      dtype=theano.config.floatX)

    # compile the functions for loss and accuracy
    train_fn = theano.function([input_var, target_var],
                               [loss, test_acc],
                               updates=updates,
                               name="train function")

    val_fn = theano.function([input_var, target_var],
                             [test_loss, test_acc],
                             name="test function")

    # Prediction
    predict_value = theano.function([input_var], test_choice,
                                    name="predict value function")


    
    results_file = open(name + "_" + movepart + ".txt", 'a')
    # print config
    if(load is False):
        initial_epoch = 0

        config_string += "\nLr\t" + str(lr_alfa0)
        if (lr_annealing):
            config_string += "\tkappa\t" + str(lr_k)
        config_string += "\nb1\t" + str(b1) + "\nb2\t" + str(b2)
        config_string += "\nep:\t" + str (num_epochs) + "\t"
        config_string += "pat:\t" + str(patience) + "\n"
        print("\nSelected configuration:\n" + config_string)
        results_file.write(config_string)

    results_file.flush()
    results = numpy.zeros((5, num_epochs))

    size = len(X_train)
    num_batch = size / batch_size
    if size % batch_size != 0:
        num_batch += 1

    # Finally, launch the training loop.
    print("Starting training...")
    print("Here we gooooo")
    print("REALLY! HERE WE GO!")
    time1 = time.time()
    bestval = 0
    bestvalepoch = initial_epoch
    save_net_weights(network, name + '_' + movepart)
    # We iterate over epochs:
        
    results_file.write("Epoch\tTrain_e\tVal_e\tTrain_a\tVal_a\tBetter?")
        
    for epoch in range(initial_epoch, num_epochs):
        better = False
        
        # early stopping
        if (bestvalepoch + patience) < epoch:
            print("\tPatience ended!")
            break

        # learning rate annealing
        if lr_annealing:
            lr = (lr_alfa0 / (1 + lr_k * epoch))

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        # train_batches = 0
        start_time = time.time()


        train_err = 0
        val_err = 0
        
        indexes = range(len(X_train))
        random.shuffle(indexes)
        
        for num in range(0, num_batch):
            first_index = batch_size*(num)
            second_index = batch_size*(num+1)
            if(num == num_batch-1):
                second_index = size
                
            X_batch = []
            y_batch = []
            
            indexes_batch = indexes[first_index:second_index]
            
            for index in indexes_batch:
                X_batch.append(X_train[index])       
                y_batch.append(y_train[index])
                
                
            parz_err, parz_acc = train_fn(X_batch, y_batch)
            l = len(y_batch)
            train_err += parz_err*l
            train_acc += parz_acc*l

        train_err = train_err/size
        train_acc = train_acc/size

        # train_err, train_acc = train_fn(X_train, y_train)

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0


        val_err, val_acc = val_fn(X_val, y_val)

        if val_acc > bestval:
            bestval = val_acc
            bestvalepoch = epoch
            better = True
    
        # val_leg = legality_fn(X_val)
        
        
        results[0][epoch] = train_err
        results[1][epoch] = train_acc
        results[2][epoch] = val_err
        results[3][epoch] = val_acc
        # results[4][epoch] = val_leg
        
        logstring = (str(epoch + 1) + "\t" + "%.6f" %
                     (train_err) + "\t" + "%.6f" %
                     (val_err) + "\t" + "%.2f" %
                     (train_acc * 100) + "\t" + "%.2f" %
                     (val_acc * 100))
        if (better):
            logstring = logstring + "\tBetter!"
        results_file.write(logstring) 

        

        results_file.write("\n")

        # Then we print the results for this epoch:
        print(time.strftime('%H:%M %d %b: ') +
              "- Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time) + 
				"\tLr: " + str(lr))
        
        if better:
            save_net_weights(network, name + '_' + movepart)
            results_file.flush()


    time2 = time.time()
    totaltime = time2-time1
    print("TIME OCCURRED: " + str(totaltime))
    results_file.write("TIME OCCURRED: " + str(totaltime) + " seconds\n")


    # After training, we compute and print the test error:
    network = load_net_weights(network, name + '_' + movepart)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    if(regularization):
        test_loss = test_loss + reg
    
    val_fn = theano.function([input_var, target_var],
                     [test_loss, test_acc],
                     name="test function")
    
    test_err = 0
    test_acc = 0

    test_err, test_acc = val_fn(X_test, y_test)
    
    
    print("Final results:")
    print("  test loss:\t\t\t" + "%.6f" % (test_err))
    print("  test accuracy:\t\t" + "%.2f" % (test_acc * 100) + " %")
    
    results_file.write("Test:\t " + "%.6f" % (test_err) + "\t" +
                       "%.6f" % (test_acc * 100))
    results_file.write("\n")
    results_file.flush();
                      
    val_err, val_acc = val_fn(X_val, y_val)
    print("  val loss:\t\t\t" + "%.6f" % (val_err))
    print("  val accuracy:\t\t" + "%.2f" % (val_acc * 100) + " %")
    
    results_file.write("Validation:\t " + "%.6f" % (val_err) + "\t" +
                       "%.6f" % (val_acc * 100))
    results_file.write("\n")
    results_file.flush();
                      
    train_err = 0
    train_acc = 0
    for num in range(0, num_batch):
        first_index = batch_size*(num)
        second_index = batch_size*(num+1)
        if(num == num_batch-1):
            second_index = size
        X_batch = X_train[first_index:second_index]
        y_batch = y_train[first_index:second_index]
        # if (masked):
        #    train_err, train_acc = train_fn(X_batch, y_batch,
        #                                    X_batch[:, 48:72])
        # else:
        parz_err, parz_acc = val_fn(X_batch, y_batch)
        l = len(y_batch)
        train_err += parz_err*l
        train_acc += parz_acc*l

    train_err = train_err/size
    train_acc = train_acc/size

    print("  train loss:\t\t\t" + "%.6f" % (train_err))
    print("  train accuracy:\t\t" + "%.2f" % (train_acc * 100) + " %")
    
    results_file.write("Training:\t " + "%.6f" % (train_err) +
                       "\t" + "%.6f" % (train_acc * 100))
    results_file.write("\n")
    results_file.flush();


    # print false predictions
    # results_file.write("\n\n\n")
    # for i in range(0, len(X_test)):
    #     num = predict_value(X_test[i:i+1])
    #     if(num!=y_test[i]):
    #         results_file.write("\nExpected: " + str(y_test[i]) + 
    #                            "\tResult: "+str(num[0]))
    # results_file.write("\n\n\n")

    # verify accuracy
    # test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                   dtype=theano.config.floatX)
    # ta_fn = theano.function([input_var, target_var], test_acc,
    #                         name="test accuracy function")
    # 
    # print(str(ta_fn(X_test, y_test)))
    
    
    results_file.close()
    sys.stdout.flush()

    return input_var, network
