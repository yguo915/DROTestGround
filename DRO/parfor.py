from multiprocessing import Pool

def func(t):
    s = 0
    for i in range(5):
        s = s+i
    print(t, s)


if __name__ == '__main__':
    pool = Pool(processes=8)
    pool.map(func, range(10))
    pool.close()
    pool.join()
    print("done")
