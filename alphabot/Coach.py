from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import multiprocessing as mp
#import threading as th
import copy
import logging


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        #self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overridden in loadTrainExamples()

    def executeEpisode(self, output):
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
        game = copy.deepcopy(self.game)
        current_game = game.getInitGame()
        mcts = MCTS(game, self.nnet, self.args)
        #print(id(game))
        curPlayer = 1 if f'{current_game.current_player}' == 'Player1' else -1
        episodeStep = 0

        while True:
            episodeStep += 1
            print('---Episode step ' + str(episodeStep) + '--- ' + mp.current_process().name) #os.getpid())
            # state = self.game.getState(self.curPlayer)
            state = game.getState(current_game)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = mcts.getActionProb(state, temp=temp)
            pi_reshape = np.reshape(pi, (21, 18))
            # sym = self.game.getSymmetries(state, pi)
            # s = self.game.getState(current_game)
            # trainExamples.append([s, self.curPlayer, pi, None])
            trainExamples.append([state, curPlayer, pi, None])
            # for b,p in sym:
            #     trainExamples.append([b, self.curPlayer, p, None])
            action = np.random.choice(len(pi), p=pi)
            a, b = np.unravel_index(np.ravel(action, np.asarray(pi).shape), pi_reshape.shape)
            # current_game, self.curPlayer = self.game.getNextState(self.curPlayer, (a[0], b[0]), current_game)
            next_state, curPlayer = game.getNextState(curPlayer, (a[0], b[0]), current_game)

            r = game.getGameEnded(current_game)

            if r!=0:
                #return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]
                output.put([(x[0],x[2],r*((-1)**(x[1]!=curPlayer))) for x in trainExamples])
                return

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
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                # Define an output queue
                output = mp.Queue()

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
    
                for eps in range(self.args.numEps):
                    #self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
                    #if __name__ == 'Coach':
                    #mp.freeze_support()
                    # Setup a list of processes that we want to run
                    processes = [mp.Process(target=self.executeEpisode, args=(output,)) for x in range(2)]
                    #processes = [th.Thread(target=self.executeEpisode, args=(output,)) for x in range(2)]
                    # Run processes
                    for p in processes:
                        p.start()
                    # Get process results from the output queue
                    iterationTrainExamples += [output.get() for p in processes]
                    #iterationTrainExamples += self.executeEpisode()
                    # Wait for all processes to terminate
                    for p in processes:
                        p.join()
                    # TODO: Don't wait for finished processes, start new ones instead after getting their results! Keep x processes alive until numEps is reached!

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)
            
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)
            
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')
            # arena = Arena(lambda x: np.where(x==np.max(pmcts.getActionProb(x, temp=0))),
            #               lambda x: np.where(x==np.max(nmcts.getActionProb(x, temp=0))), self.game)
            arena = Arena(lambda x: pmcts.getActionProb(x, temp=0),
                          lambda x: nmcts.getActionProb(x, temp=0), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
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
