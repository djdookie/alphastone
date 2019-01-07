import os, sys
from pickle import Pickler, Unpickler


def saveTrainExamples(trainExamples, folder, filename):
    #folder = self.args.checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    #examplesFile = filepath + ".examples"
        
    #filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
    with open(filepath, "wb+") as f:
        Pickler(f).dump(trainExamples)
    f.closed

def loadTrainExamples(folder, filename):
    filepath = os.path.join(folder, filename)
    #examplesFile = filepath + ".examples"
    if not os.path.isfile(filepath):
        print(filepath)
        r = input("File with trainExamples not found. Continue? [y|n]")
        if r != "y":
            sys.exit()
    else:
        print("File with trainExamples found. Read it.")
        with open(filepath, "rb") as f:
            trainExamples = Unpickler(f).load()
        f.closed
        return trainExamples

if __name__=="__main__":
    trainExamples1 = loadTrainExamples('./temp/','merged.pth.tar.examples')
    count = [len(x) for x in trainExamples1]
    print("Read", str(len(trainExamples1)), "iterations with", sum(count), "examples")

    trainExamples2 = loadTrainExamples('./temp/','checkpoint_1.pth.tar.examples')
    count = [len(x) for x in trainExamples2]
    print("Read", str(len(trainExamples2)), "iterations with", sum(count), "examples")

    trainExamplesMerged = []
    trainExamplesMerged += trainExamples1
    trainExamplesMerged += trainExamples2

    saveTrainExamples(trainExamplesMerged, './temp/','best.pth.tar.examples')
    count = [len(x) for x in trainExamplesMerged]
    print("Saved", str(len(trainExamplesMerged)), "iterations with", sum(count), "examples")

