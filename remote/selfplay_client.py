from Game import YEET as Game
from NNet import NNetWrapper as nn
from dotted_dict import DottedDict as dotdict
from multiprocessing import freeze_support
import os, logging, copy, time, sys
from collections import deque
from MCTS import MCTS
import numpy as np
from utils import Bar, AverageMeter
from pickle import Pickler, Unpickler
from random import shuffle
import multiprocessing as mp
from multiprocessing import current_process
from multiprocessing.managers import BaseManager
from concurrent.futures import ProcessPoolExecutor
import tqdm
import functools

class QueueManager(BaseManager): pass

class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__()  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overridden in loadTrainExamples()

    def executeEpisode(self, x):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (state,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        logger = logging.getLogger("fireplace")
        logger.setLevel(logging.WARNING)
        trainExamples = []
        #print(id(self.game))
        #game = copy.deepcopy(self.game)
        current_game = self.game.getInitGame()
        #self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree -> not needed since we get a copy of self.mcts in this child process, and MCTS itself uses a deep clone of this copy
        #curPlayer = 1 if f'{current_game.current_player}' == 'Player1' else -1
        #self.curPlayer = 1 if f'{current_game.current_player}' == 'Player1' else -1
        self.curPlayer = 1 if current_game.current_player.name == 'Player1' else -1
        #print(id(self.curPlayer))
        episodeStep = 0

        while True:
            episodeStep += 1
            #print('---Episode step ' + str(episodeStep) + '--- ' + current_process().name) #os.getpid())
            state = self.game.getState(current_game)                    # state is from the current player's perspective
            temp = int(episodeStep < self.args.tempThreshold)

            # TODO: No MCTS reset? Should work because player switch leads to new mirror state and info in tree could possibly be reused. Otherwise starting a new tree could speed things up here (faster lookups through smaller mcts lists)!
            pi = self.mcts.getActionProb(state, temp=temp)              # pi is from the current player's perspective
            # print(self.mcts)
            pi_reshape = np.reshape(pi, (21, 18))
            # sym = self.game.getSymmetries(state, pi)
            trainExamples.append([state, self.curPlayer, pi, None])     # TODO: Is None still needed?
            # for b,p in sym:
            #     trainExamples.append([b, self.curPlayer, p, None])
            action = np.random.choice(len(pi), p=pi)
            a, b = np.unravel_index(np.ravel(action, np.asarray(pi).shape), pi_reshape.shape)
            next_state, self.curPlayer = self.game.getNextState(self.curPlayer, (a[0], b[0]), current_game)

            r = self.game.getGameEnded(current_game, self.curPlayer)    # returns 0 if game has not ended, 1 if curPlayer won, -1 if curPlayer lost

            if r!=0:
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')

            # Find and load newest model (name scheme 'x.pth.tar' with highest x)
            modelfile = self.get_newest_model()
            print("Loading newest model:", modelfile)
            nnet.load_checkpoint(args.modelspath, modelfile)
            
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
 
                with ProcessPoolExecutor(self.args.numThreads) as executor:
                    #self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree => not needed because we get a new instance per process start
                    # Setup a list of processes that we want to run
                    #results = list(tqdm.tqdm(executor.map(self.executeEpisode, range(self.args.numEps)), total=self.args.numEps, desc='Self-play matches'))
                    for result in list(tqdm.tqdm(executor.map(self.executeEpisode, range(self.args.numEps)), total=self.args.numEps, desc='Self-play matches')):
                        iterationTrainExamples += result

                #iterationTrainExamples = [r for r in results]

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            # if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            #     print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
            #     self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(modelfile, i-1)
            
    def getCheckpointFile(self, modelfile, iteration):
        return modelfile + '_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, modelfile, iteration):
        folder = self.args.examplespath
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(modelfile, iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def list_files(self, directory, extension):
        return (f for f in os.listdir(directory) if f.endswith('.' + extension))

    def get_modelnumber(self, filename):
        file_name = os.path.basename(filename)
        index_of_dot = file_name.index('.')
        file_name_without_extension = file_name[:index_of_dot]
        return int(file_name_without_extension)

    def get_newest_model(self):
        files = self.list_files(args.modelspath, "pth.tar")
        modelnumbers = list(map(self.get_modelnumber, files))
        maximum = max(modelnumbers)
        return str(maximum) + ".pth.tar"

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,    # degree of exploration in MCTS.getActionProb(). switch from temperature=1 to temperature=0 after this episode step
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,      # 25    # TODO: much more sims needed?
    'cpuct': 2,             # degree of exploration for upper confidence bound in MCTS.search() => TODO: try 2?

    'modelspath': './models/',
    'examplespath': './examples/',
    # 'numItersForTrainExamplesHistory': 100000,      #20
    'numThreads': 2,
})

if __name__=="__main__":
    #freeze_support()
    g = Game(is_basic=True)
    # Suppress logging from fireplace
    logger = logging.getLogger("fireplace")
    logger.setLevel(logging.WARNING)

    nnet = nn()

    c = Coach(g, nnet, args)
    c.learn()