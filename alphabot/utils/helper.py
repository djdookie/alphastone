from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def parallel_process(function, array, workers=8, use_kwargs=False, front_num=0, desc=None):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            workers (int, default=8): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
            desc (str): Prefix for the tqdm progressbar.
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    front = []
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set workers to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if workers==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=workers) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': False,
            'leave': True,
            'desc': desc
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    # for i, future in tqdm(enumerate(futures)):
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out