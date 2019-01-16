from multiprocessing.managers import BaseManager
from pickle import Pickler, Unpickler, loads
from NNet import NNetWrapper as nn
from dotted_dict import DottedDict as dotdict
from random import shuffle
import time, os, sys
import torch
from sklearn.model_selection import train_test_split
#from tensorboard_logger import configure, log_value
from tensorboardX import SummaryWriter

args = dotdict({
    'modelspath': './models/',
    'examplespath': './examples/',
    'epochs': 4,       # best 25,
    'validation': False
})

class QueueManager(BaseManager): pass

class TrainingClient:
    def __init__(self):
        # self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.nnet = nn()

    def loadTrainExamples(self):
        # modelFile = os.path.join(args.examplespath, 'best.pth.tar')
        # examplesFile = modelFile+".examples"
        examplesFile = os.path.join(args.examplespath, 'best.pth.tar.examples')
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
    # print("Load current neural network")
    # client.nnet.load_checkpoint(folder=client.args.modelspath, filename='0b.pth.tar')

    # load trainexamples
    print("Load trainExamples from file")
    trainExamplesHistory = client.loadTrainExamples()

    # shuffle examples before training      => possibly not needed because network takes random samples for training
    trainExamples = []
    for e in trainExamplesHistory:
        trainExamples.extend(e)
    shuffle(trainExamples)
  
    # split training and test sets
    examples_train, examples_test = train_test_split(trainExamples, test_size=0.25)

    # configure logger
    #configure("logs/run-1", flush_secs=5)
    training_writer = SummaryWriter('runs/training-2')
    test_writer = SummaryWriter('runs/test-2')
    dif_writer = SummaryWriter('runs/dif-2')
    max_loss_pi_dif = float('-inf')

    # train neural network
    for epoch in range(args.epochs):
        print('EPOCH ::: ' + str(epoch+1))
        loss_pi_train, loss_v_train = client.nnet.train(examples_train)
        if args.validation:
            # configure("logs/run-1/training", flush_secs=5)
            # log_value('training_loss_pi', l_pi, epoch+1)
            # log_value('training_loss_v', l_v, epoch+1)
            # data grouping by `slash`
            training_writer.add_scalar('loss_pi', loss_pi_train, epoch+1)
            training_writer.add_scalar('loss_v', loss_v_train, epoch+1)

            loss_pi_test, loss_v_test = client.nnet.test(examples_test)
            # configure("logs/run-1/test", flush_secs=5)
            test_writer.add_scalar('loss_pi', loss_pi_test, epoch+1)
            test_writer.add_scalar('loss_v', loss_v_test, epoch+1)

            loss_pi_dif = loss_pi_test - loss_pi_train
            loss_v_dif = loss_v_test - loss_v_train
            dif_writer.add_scalar('dif_pi', loss_pi_dif, epoch+1)
            dif_writer.add_scalar('dif_v', loss_v_dif, epoch+1)

            # Early stopping
            if max_loss_pi_dif < loss_pi_dif:
                max_loss_pi_dif = loss_pi_dif
                if max_loss_pi_dif > 0:
                    print('Early stopping because difference in loss_pi between training and test is positive and growing.')
                    break

    print("Save new trained neural network")
    client.nnet.save_checkpoint(folder=args.modelspath, filename='best.pth.tar')

    # print('exiting...')