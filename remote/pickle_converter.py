from pickle import Pickler, Unpickler, loads
import pickle

# needs numpy 1.16.1 to be able to read those older pickle files because of this bug: https://github.com/numpy/numpy/pull/12842

file_origin = './examples/0.pth.tar_6.examples'
file_target = './examples.con/0.pth.tar_6.examples'

with open(file_origin, "rb") as f:
    result = Unpickler(f).load()

with open(file_target, "wb+") as f:
    Pickler(f, protocol=pickle.HIGHEST_PROTOCOL).dump(result)
