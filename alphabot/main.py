from Coach import Coach
from Game import YEET as Game
from NNet import NNetWrapper as nn
#from utils import dotdict
from dotted_dict import DottedDict as dotdict
#from multiprocessing import freeze_support
import multiprocessing as mp
import logging, psutil, os, sys

args = dotdict({
    'numIters': 100,
    'numEps': 100,
    'tempThreshold': 15,    # degree of exploration in MCTS.getActionProb(). switch from temperature=1 to temperature=0 after this episode step
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,      # 25    # TODO: much more sims needed?
    'arenaCompare': 40,
    'cpuct': 2,             # degree of exploration for upper confidence bound in MCTS.search() => TODO: try 2?

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 50,      #20
    'numThreads': 2,
    'remoteTraining': True
})

if __name__=="__main__":
    #freeze_support()
    # Start processes with lower priority to prevent system overload/hangs/freezes. Also set multiprocessing start method to spawn for Linux, since forking makes trouble
    p = psutil.Process(os.getpid())
    if sys.platform.startswith('win32'):
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    elif sys.platform.startswith('linux'):
        p.nice(5)
        mp.set_start_method('spawn')

    g = Game(is_basic=True)
    # Suppress logging from fireplace
    logger = logging.getLogger("fireplace")
    logger.setLevel(logging.WARNING)

    nnet = nn()

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
