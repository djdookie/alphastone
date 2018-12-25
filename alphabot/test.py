import multiprocessing as mp
import random
import string
#from utils import dotdict
from dotted_dict import DottedDict as dotdict
import time

dd = dotdict({
    'numIters': 100,
    'numThreads': 4,
})

# define a example function
def rand_string(length, dd, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(random.choice(
                        string.ascii_lowercase
                        + string.ascii_uppercase
                        + string.digits)
                   for i in range(length))
    #time.sleep(10)
    output.put(rand_str)

if __name__ == '__main__':
    random.seed(123)

    # Define an output queue
    output = mp.Queue()

    # Setup a list of processes that we want to run
    processes = [mp.Process(target=rand_string, args=(5, dd, output)) for x in range(4)]

    # Run processes
    for p in processes:
        p.start()

    # Get process results from the output queue -> we wait for each of 4 process results here
    results = [output.get() for p in processes]

    # Exit the completed processes -> wait for the 4 processes to terminate
    # (a process that has put items in a queue will wait before terminating until all the buffered items are fed by the “feeder” thread to the underlying pipe)
    # https://docs.python.org/3.7/library/multiprocessing.html#all-start-methods -> "Joining processes that use queues"
    for p in processes:
        p.join()

    print(results)
