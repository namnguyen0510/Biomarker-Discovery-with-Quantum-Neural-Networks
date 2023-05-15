import evaluator
from multiprocessing import Pool
import numpy as np


num_workers = 10

if __name__ == '__main__':
    with Pool(num_workers) as p:
        print(p.map(evaluator.main, np.random.randint(0,999, num_workers)))
