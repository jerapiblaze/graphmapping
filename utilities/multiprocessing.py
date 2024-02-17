import multiprocessing as mp

def MultiProcessing(Worker, args: tuple, n_worker: int):
    print(f"MultiProcess: {n_worker}")
    ps = []
    for i in range(n_worker):
        p = mp.Process(
            target=Worker,
            args=args
        )
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

def IterToQueue(iter:list) -> mp.Queue:
    q = mp.Queue()
    for item in iter:
        q.put(item)
    return q