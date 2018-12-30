from multiprocessing.managers import BaseManager
from dotted_dict import DottedDict as dotdict
from NNet import NNetWrapper as nn
from pickle import Pickler, Unpickler
import time, os, sys
import torch
import io


class QueueManager(BaseManager): pass

class Client:
    args = dotdict({
        'load_folder_file': ('../alphabot/temp.old/','checkpoint_1.pth.tar'),
        'checkpoint': '../alphabot/temp/'
    })

    def __init__(self):
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.nnet = nn()

    # def getCheckpointFile(self, iteration):
    #     return 'checkpoint_' + str(iteration) + '.pth.tar'

    # def saveTrainExamples(self, iteration):
    #     folder = self.args.checkpoint
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    #     filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
    #     with open(filename, "wb+") as f:
    #         Pickler(f).dump(self.trainExamplesHistory)
    #     f.closed

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
                #self.trainExamplesHistory = Unpickler(f).load()
                self.trainExamplesHistory = f.read()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def loadModelFile(self):
        modelFile = os.path.join(self.args.checkpoint, 'temp.pth.tar')
        if not os.path.isfile(modelFile):
            print(modelFile)
            r = input("Model file not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with modelFile found. Read it.")
            with open(modelFile, "rb") as f:
                #self.trainExamplesHistory = Unpickler(f).load()
                result = f.read()
            f.closed
            return result


if __name__=="__main__":
    print('starting...')
    client = Client()

    QueueManager.register('job_queue')
    QueueManager.register('result_queue')
    m = QueueManager(address=('localhost', 50000), authkey=b'thisismysecret')
    m.connect()
    job_queue = m.job_queue()
    result_queue = m.result_queue()

    # print("Load neural network")
    # client.nnet.load_checkpoint(folder=client.args.checkpoint, filename='temp.pth.tar')
    #client.nnet.save_checkpoint(folder=client.args.checkpoint, filename='temp2.pth.tar')
    #modelfile = client.loadModelFile()
    # inmemoryfile = io.BytesIO()
    #torch.save(client.nnet.nnet.state_dict(), inmemoryfile)
    # torch.save({'state_dict' : client.nnet.nnet.state_dict()}, inmemoryfile)

    # checkpoint = torch.load(inmemoryfile)
    # client.nnet.nnet.load_state_dict(checkpoint['state_dict'])

    # if args.load_model:
    print("Load trainExamples from file")
    client.loadTrainExamples()

    # Create job (dictionary)
    #job = {"neuralnet" : modelfile, "examples" : client.trainExamplesHistory}
    job = {"examples" : client.trainExamplesHistory}

    print(job_queue.qsize())
    print("sending...")
    # job_queue.put(client.trainExamplesHistory)
    job_queue.put(job)

    print("receiving...")
    #buffer = result_queue.get()
    result = result_queue.get()
    # checkpoint = torch.load(buffer)
    #client.nnet.nnet.load_state_dict(checkpoint['state_dict'])

    if result['finished'] == True:
        client.nnet.load_checkpoint(folder=client.args.checkpoint, filename='temp.pth.tar')

    # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
    # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

    print('exiting...')