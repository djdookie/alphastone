from multiprocessing.managers import BaseManager

class QueueManager(BaseManager): pass

QueueManager.register('job_queue')
QueueManager.register('result_queue')
m = QueueManager(address=('localhost', 50000), authkey=b'thisismysecret')
m.connect()
job_queue = m.job_queue()
result_queue = m.result_queue()

job_queue.put(3)
print(result_queue.get())