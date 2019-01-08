from multiprocessing.managers import BaseManager
from pickle import Pickler, Unpickler, loads
from NNet import NNetWrapper as nn
from dotted_dict import DottedDict as dotdict
from random import shuffle
import time, os, sys
import torch

class QueueManager(BaseManager): pass

class TrainingClient:
    args = dotdict({
        'modelspath': './models/',
        'examplespath': './examples/'
    })

    def __init__(self):
        # self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.nnet = nn()

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.examplespath, 'best.pth.tar')
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                return Unpickler(f).load()
            f.closed

if __name__=="__main__":
    # print('starting...')
    client = TrainingClient()

    # load neural network
    print("Load current neural network")
    client.nnet.load_checkpoint(folder=client.args.modelspath, filename='0.pth.tar')

    # load trainexamples
    print("Load trainExamples from file")
    trainExamplesHistory = client.loadTrainExamples()

    # shuffle examples before training      => possibly not needed because network takes random samples for training
    trainExamples = []
    for e in trainExamplesHistory:
        trainExamples.extend(e)
    shuffle(trainExamples)
  
    # train neural network
    client.nnet.train(trainExamples)

    print("Save new trained neural network")
    client.nnet.save_checkpoint(folder=client.args.modelspath, filename='0a.pth.tar')

    # print('exiting...')