# Neural Nine Men's Morris

Neural Nine Men's Morris (NNMM) is a software capable to build, train, test and use Deep Neural Networks which can be used for playing the game of Nine Men's Morris.
It has been demonstrated that the system is capable to learn to play by the rules of the game, even if the knowledge of those rules has not been provided to it.
It has been designed by Andrea Galassi, as part of his master thesis in Computer Science Engineering ("Ingegneria Informatica", in italian), and it has been improved as part of a successive work.
Please feel free to contact him (a.galassi *at* unibo.it) or his thesis supervisors (Paola Mello, paola.mello *at* unibo.it, Federico Chesani federico.chesani *at* unibo.it) for further questions.

The result of this work have been published in IEEE Transactions on Games: https://doi.org/10.1109/TG.2018.2804039
Further information can be found on the software website: http://ai.unibo.it/NNMM



# Implementation

The system has been written in Python language, relying on the Lasagne, Theano and Numpy libraries.


# License

The software is under MIT license. Further details can be found in the file LICENSE.txt


# Documentation (V 1.1.1)

Executable files:
	
	main_dl.py
		Analyse a dataset to collect statistics about the legal moves for each states
    
	main_da.py
	        Analyse a dataset to collect statistics about the state composition
		
	main_play.py
		Client that can be used to play the game of Nine Men's Morris.
		Communicate thought socket on port 5802 for white player and 5803 for black player.
		A state string is expected as input on the port and a move string is sent back with format "TO[FROM][REMOVE]".
	
	main_reliability.py
		Execute a reliability test: it measure the ability of the system to assign an higher score to legal choices.
	
	main_test.py
		Execute a legality and accuracy test: it measures the ability of the system to provide accurate moves (equal to the ones in the dataset) abd legal moves.
	
	main_training.py
		Allows to build a network from scratch or to load an existing one and to train it.
		Networks characteristics must be defined modifying the file.


Source files:

	dataprocessing.py
		Contains the functions used to manipulate the data.
		
	legality.py
		Contains the functions used to establish the legality of a move.
		
	networks.py
		Contains the functions used to create, load, save and use neural networks.
		
	testing.py
		Contains the functions used to test the system according to different characteristics.
		
	training.py
		Contains the functions used to train a neural network system.
