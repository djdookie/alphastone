from multiprocessing.managers import BaseManager
from pickle import Pickler, loads
from NNet import NNetWrapper as nn
from dotted_dict import DottedDict as dotdict
from random import shuffle
import time, os, sys
import torch

class QueueManager(BaseManager): pass

class TrainingClient:
    args = dotdict({
        # 'load_folder_file': ('../alphabot/temp/','checkpoint_0.pth.tar'),
        'checkpoint': '../alphabot/temp/'
    })

    def __init__(self):
        # self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.nnet = nn()

    # def unpickle(self, file):
    #     #return Unpickler(file).load()
    #     return loads(file)

    # def saveModelFile(self, file):
    #     folder = self.args.checkpoint
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    #     filename = os.path.join(folder, "temp.pth.tar")
    #     with open(filename, "wb+") as f:
    #         Pickler(f, protocol=pickle.HIGHEST_PROTOCOL).dump(file)
    #     f.closed

if __name__=="__main__":
    print('starting...')
    client = TrainingClient()

    QueueManager.register('job_queue')
    QueueManager.register('result_queue')
    m = QueueManager(address=('localhost', 50000), authkey=b'thisismysecret')
    m.connect()
    job_queue = m.job_queue()
    result_queue = m.result_queue()

    # # initially load neural network if exists
    # client.nnet.load_checkpoint(folder=client.args.checkpoint, filename='temp.pth.tar')

    while True:
        # receive job (neural network and train examples)
        #print(job_queue.qsize())
        print("receiving...")
        job = job_queue.get()

        #trainExamples = client.unpickle(job)
        #modelfile = job['neuralnet']
        #client.saveModelFile(modelfile)
        #trainExamples = loads(job['examples'])
        trainExamples = job['examples']

        # shuffle examples before training
        # trainExamples = []
        # for e in trainExamples:
        #     trainExamples.extend(e)
        # shuffle(trainExamples)

        # load neural network
        print("Load current neural network")
        client.nnet.load_checkpoint(folder=client.args.checkpoint, filename='temp.pth.tar')
        
        # train neural network
        client.nnet.train(trainExamples)

        print("Save new trained neural network")
        client.nnet.save_checkpoint(folder=client.args.checkpoint, filename='remote.pth.tar')

        # buffer = io.BytesIO()
        # torch.save({
        #     'state_dict' : client.nnet.nnet.state_dict()
        # }, buffer)

        # Wait some time
        time.sleep(5)

        result = {"finished" : True}

        print("sending...")
        # result_queue.put(buffer)
        result_queue.put(result)

    print('exiting...')