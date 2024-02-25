from tqdm import tqdm
import time
# create 1000*8192*4*3 numpy array randomly
import numpy as np
import os

random_array = np.random.rand(1000, 8192, 4, 3)
np.savez_compressed("test_compressed", random_array = random_array)
