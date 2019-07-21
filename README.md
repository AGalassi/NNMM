If you use this software as part of any work, please cite it as:
	
	F. Chesani, A. Galassi, M. Lippi and P. Mello, "Can Deep Networks Learn to Play by the Rules? A Case Study on Nine Men's Morris," in IEEE Transactions on Games, vol. 10, no. 4, pp. 344-353, Dec. 2018. doi: 10.1109/TG.2018.2804039

### Neural Nine Men's Morris
Neural Nine Men's Morris (NNMM) is a software capable to build, train, test and use Deep Neural Networks which can be used for playing the game of Nine Men's Morris. It has been demonstrated that the system is capable to learn to play by the rules of the game, even if the knowledge of those rules has not been provided to it. It has been designed by Andrea Galassi, as part of his master thesis in Computer Science Engineering ("Ingegneria Informatica", in italian), and it has been improved as part of a successive work. Please feel free to contact him (a.galassi at unibo.it) or his thesis supervisors (Paola Mello, paola.mello at unibo.it, Federico Chesani federico.chesani at unibo.it) for further questions.

The results of this work have been published in IEEE Transactions on Games: https://doi.org/10.1109/TG.2018.2804039

Further information can be found on the software website: http://ai.unibo.it/NNMM

### Implementation
The system has been written in Python language, relying on the Lasagne, Theano and Numpy libraries.

### License
The software is under MIT license. Further details can be found in the file LICENSE.txt

### Documentation (V 1.1.1)
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
		
### Datasets
#### Good Moves Dataset (Matches Dataset)
The dataset consist of 100,154 game states and as many good moves elaborated by an Artificial Intelligence for the game of Nine Men's Morris.

None of the states in the dataset is symmetric to any other, therefore anyone can handle the symmetries as he/she prefers.
If all the symmetric states are explored, the dataset can reach 1,628,673 pairs.

The dataset contains states both reachable and unreachable during a normal match, decreasing the probability of reaching a training state during a testing match. The moves contained in it could be different from the optimal one, however, it constitutes a good knowledge base, from which other AI system can learn to play the game.

All the data have been generated making play an Artificial Intelligence called Deep Mill against other artificial intelligence and gathering the choices made by Deep Mill during the games.

Three version of the dataset are available:

- The COMPLETE DATASET contains all the data, without the symmetric pairs.
- The GAMING DATASET do not contains data coming from a regular match starting with an empty board. Therefore it contains states which are more unlikely to be reached during a match.
- The EXPANDED DATASET contains all the data and the symmetric pairs.


#### Reachable States Dataset
The dataset consist of 2,085,613 states which are reachable through a finite sequence of legal moves starting from the initial empty board configurations. It has been generated exploring the space of the game states applying random choices from a reachable configurations.

None of the states contained in this dataset is present in the Good Moves Dataset.

- The COMPLETE STATES DATASET contains all the states, without the symmetric ones.
- The EXTENDED STATES DATASET contains all the states. with the symmetric ones.
