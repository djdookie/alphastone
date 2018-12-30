from multiprocessing.managers import BaseManager

class QueueManager(BaseManager): pass

QueueManager.register('job_queue')
QueueManager.register('result_queue')
m = QueueManager(address=('localhost', 50000), authkey=b'thisismysecret')
m.connect()
job_queue = m.job_queue()
result_queue = m.result_queue()

def cube(x):
    return x**3

while True:
    print("ping")
    job = job_queue.get()
    print(job)
    result = cube(job)
    result_queue.put(result)
