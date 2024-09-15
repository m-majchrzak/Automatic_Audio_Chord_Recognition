import os
import math
from multiprocessing import Pool


def test_func(list):
    pid = os.getpid()
    for i in list:
        print('Number:', i, ' PID:', pid)


midi_paths = [i for i in range(50)]
num_workers=20
num_path_per_process = math.ceil(len(midi_paths) / num_workers)
args = [midi_paths[i * num_path_per_process:(i + 1) * num_path_per_process]
                            for i in range(num_workers)]

p = Pool(processes=num_workers)
p.map(test_func, args)
p.close()
