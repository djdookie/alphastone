import subprocess
import time
# p = subprocess.Popen('ping 127.0.0.1')
# subprocess.run('ping 127.0.0.1')

for i in range(5):
    subprocess.Popen('python selfplay_client.py')
    time.sleep(1)