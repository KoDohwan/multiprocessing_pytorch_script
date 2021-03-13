import time
from multiprocessing import Pool, current_process

def main_worker(items):
    i, j = items
    for k in range(i * j):
        a = 10
    return True

def main(config):
    num_workers = config['num_workers']

    iteration = zip(range(1, 2000, 1), range(2000, 1, -1))

    pool = Pool(num_workers)
    pool.map(main_worker, iteration)

if __name__ == '__main__':
    config = {'num_workers': 200}
    start = time.time()
    main(config)
    end = time.time()
    print(f'{end - start:.4f}s')