from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from operator import mul
from ..persist import *
from .. import *

plt.ion()
fig, ax = plt.subplots(1, 3, figsize=(13,4))
plt.tight_layout(); plt.subplots_adjust(wspace=0.1)

class PersistencePlot(DioPersist):
    def __init__(self, fun, dset, x):
        self.fun, self.x, self.shape = fun, x, SHAPE[dset]
    def initialize(self):
        self.plot_dgm(ax[2])
        self.dgm = self.np_dgm()
        self.dims = range(1, len(self.D)) if self.fun != 'circular' else [1]
        self.kd = {d : KDTree(self.dgm[d]) for d in self.dims if len(self.dgm[d]) > 0}
        self.cache, self.last = {}, None
    def query(self, e):
        self.last = e
        p = np.array([e.xdata, e.ydata])
        if e.inaxes == ax[2]:
            ds = [(k, v.query(p)) for k, v in self.kd.iteritems()]
            dim, res = min(ds, key=lambda x: x[1][0])
            pt = self.D[dim][res[1]]
            self.plot_dgm(ax[2])
            b, d = self.dgm[dim][res[1]]
            ax[2].scatter(b, d, s=20, zorder=2, c='black')
            return pt
        return None
    def toimg(self, v):
        z = v.reshape(-1, *reversed(self.shape)).sum(axis=0)
        return normalize(np.swapaxes(np.stack(z, axis=0).T, 0, 1))
    def plot_chain(self, key):
        arr = np.zeros(reduce(mul, self.shape, 1), dtype=float)
        arr[[v for s in self.cache[key] for v in s]] = 1.
        img = self.toimg(arr)
        if len(self.shape) > 2:
            a = img.sum(axis=2)
            a[a > 0] = 1.
            mask = np.concatenate([img, a[..., np.newaxis]], axis=2)
        else:
            mask = np.stack([img] * 4, axis=2)
        map(lambda a: a.cla(), ax[:2])
        ax[0].imshow(self.x)
        ax[0].imshow(mask)
        ax[1].imshow(img)


class HomologyPlot(StarHomology, PersistencePlot):
    def __init__(self, dset, x, prime):
        PersistencePlot.__init__(self, 'homology', dset, x)
        StarHomology.__init__(self, x, prime)
        self.initialize()
    def get_cycle(self, pt):
        return self.np_cycle(self.cycle(pt))
    def plot_image(self, key):
        if not key in self.cache:
            self.cache[key] = self.get_cycle(key)
        self.plot_chain(key)

class CohomologyPlot(StarCohomology, PersistencePlot):
    def __init__(self, dset, x, prime):
        PersistencePlot.__init__(self, 'cohomology', dset, x)
        StarCohomology.__init__(self, x, prime)
        self.initialize()
    def get_cocycle(self, pt):
        return self.np_cycle(self.cocycle(pt))
    def plot_image(self, key):
        if not key in self.cache:
            self.cache[key] = self.get_cocycle(key)
        self.plot_chain(key)

class CircularPlot(StarCohomology, PersistencePlot):
    def __init__(self, dset, x, prime):
        PersistencePlot.__init__(self, 'circular', dset, x)
        StarCohomology.__init__(self, x, prime if prime != 2 else 11)
        self.initialize()
    def get_coords(self, pt):
        v = self.coords(pt)
        return self.toimg(v)
    def plot_image(self, key):
        if not key in self.cache:
            self.cache[key] = self.get_coords(key)
        ax[1].cla()
        ax[1].imshow(self.cache[key])

CLASS = {'homology' : HomologyPlot,
        'cohomology' : CohomologyPlot,
        'circular' : CircularPlot}

def anevent(x, fun='homology', prime=2, dset='cifar'):
    ax[0].imshow(x)
    OBJ = CLASS[fun](dset, x, prime)
    def event(e):
        key = OBJ.query(e)
        if key:
            OBJ.plot_image(key)
            plt.show()
    return event, OBJ

def addevent(f, *args, **kwargs):
    fun, obj = f(*args, **kwargs)
    return fig.canvas.mpl_connect('button_release_event', fun), obj
