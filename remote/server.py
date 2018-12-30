from multiprocessing.managers import BaseManager
from queue import Queue

job_queue = Queue()
result_queue = Queue()
class QueueManager(BaseManager): pass

QueueManager.register('job_queue', callable=lambda:job_queue)
QueueManager.register('result_queue', callable=lambda:result_queue)
m = QueueManager(address=('', 50000), authkey=b'thisismysecret')
s = m.get_server()
s.serve_forever()