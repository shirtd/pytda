from torchvision import datasets, transforms
from multiprocessing import Pool
from functools import partial
import numpy.linalg as la
from tqdm import tqdm
import numpy as np
import torch

'''''''''''''''
'''' DATA '''''
'''''''''''''''

SHAPE = {'mnist' : (28, 28),
        'cifar' : (32, 32, 3)}

DATA = {'mnist' : datasets.MNIST,
        'cifar' : datasets.CIFAR10}

def load_data(dset='mnist', train=True):
    data = DATA[dset]('../data', train=train, transform=transforms.ToTensor())
    X = torch.as_tensor(data.train_data if train else data.test_data, dtype=torch.float)
    y = data.train_labels if train else data.test_labels
    return X / X.max(), y

def group_data(X, y, n=-1, k=1):
    CLASS = np.unique(y)
    D = {c : X[filter(lambda i: y[i] == c, range(len(y)))][:n] for c in CLASS}
    return {c : np.array_split(D[c], k) for c in CLASS} if k != 1 else D

def circle(n=20, r=1., uniform=False, noise=0.1):
    t = np.linspace(0, 1, n, False) if uniform else np.random.rand(n)
    e = r * (1 + noise * (2 * np.random.rand(n) - 1)) if noise else r
    return np.array([e*np.cos(2 * np.pi * t),
                    e*np.sin(2*np.pi*t)]).T.astype(np.float32)

def double_circle(n=50, r=(1., 0.7), *args, **kwargs):
    p1 = circle(int(n * r[0] / sum(r)), r[0], *args, **kwargs)
    p2 = circle(int(n * r[1] / sum(r)), r[1], *args, **kwargs)
    return np.vstack([p1 - np.array([r[0], 0.]),
                    p2 + np.array([r[1], 0.])])

def torus(n=1000, R=0.7, r=0.25):
    t = 2*np.pi * np.random.rand(2, n)
    x = (R + r * np.cos(t[1])) * np.cos(t[0])
    y = (R + r * np.cos(t[1])) * np.sin(t[0])
    z = r * np.sin(t[1])
    return np.vstack([x, y, z]).T


'''''''''''''''
'' CORE UTIL ''
'''''''''''''''

def distance_matrix(X, axis=2, **kwargs):
    return la.norm(X[np.newaxis] - X[:, np.newaxis], axis=axis, **kwargs)

def normalize(x):
    return (x - x.min()) / ((x.max() - x.min()) if x.max() > 0 else 1.)

def save_pkl(fname, x):
    with open(fname, 'w') as f:
        pkl.dump(x, f)
    return fname

def load_pkl(fname):
    with open(fname, 'r') as f:
        x = pkl.load(f)
    return x

def wait(s=''):
    ret = raw_input(s)
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')
    return ret


''''''''''''''''''
''' OTHER UTIL '''
''''''''''''''''''

def pmap(fun, x, *args):
    pool = Pool()
    f = partial(fun, *args)
    try:
        y = pool.map(f, tqdm(x))
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y

def cmap(funct, x, *args):
    fun = partial(funct, *args)
    f = partial(map, fun)
    pool = Pool()
    try:
        y = pool.map(f, tqdm(x))
    except KeyboardInterrupt as e:
        print(e)
        pool.close()
        pool.join()
        sys.exit()
    pool.close()
    pool.join()
    return y
