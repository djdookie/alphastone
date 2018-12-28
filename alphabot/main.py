from Coach import Coach
from Game import YEET as Game
from NNet import NNetWrapper as nn
#from utils import dotdict
from dotted_dict import DottedDict as dotdict
import logging
import multiprocessing as mp

args = dotdict({
    'numIters': 100,
    'numEps': 4,
    'tempThreshold': 15,    # degree of exploration in MCTS.getActionProb(). switch from temperature=1 to temperature=0 after this episode step
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,             # degree of exploration for upper confidence bound in MCTS.search()

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./models/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'numThreads': 2,

})

if __name__=="__main__":
    mp.freeze_support()
    g = Game(is_basic=True)
    # Suppress logging from fireplace for performance reasons
    logger = logging.getLogger("fireplace")
    logger.setLevel(logging.WARNING)

    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
