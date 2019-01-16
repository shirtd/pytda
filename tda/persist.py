from matplotlib.colors import Normalize
import matplotlib.cm as cm
import dionysus as dio
import numpy as np

def lfilt(f, F):
    return dio.Filtration(filter(f, F))

class DioWrap:
    def __init__(self, ffun, pfun):
        self.ffun, self.pfun = ffun, pfun
    def build_filt(self, *args):
        self.F = self.ffun(*args)
    def build_pers(self, *args):
        self.H = self.pfun(self.F, *args)
        self.D = dio.init_diagrams(self.H, self.F)
    def add(self, s):
        return self.F.add(s)
    def append(self, s):
        return self.F.append(s)
    def index(self, s):
        return self.F.index(s)
    def rearrange(self, I):
        return self.F.rearrange(I)
    def sort(self):
        return self.F.sort()
    def __contains__(self, s):
        return s in self.F
    def pair(self, pt):
        return self.H.pair(pt)
    def unpaired(self):
        return self.H.unpaired()
    def np_filt(self, dim):
        return np.array([list(s) for s in self.F if s.dimension() == dim])
    def dgm_dim(self, dim):
        return sorted(self.D[dim], key=lambda p: p.death - p.birth, reverse=True)
    def get_point(self, dim, i=0):
        return self.dgm_dim(dim)[i]
    def np_dgm_dim(self, dim):
        return np.array([[p.birth, p.death] for p in self.dgm_dim(dim)])
    def np_dgm(self):
        return [self.np_dgm_dim(dim) for dim in range(len(self.D))]
    def dgm_thresh(self):
        deaths = filter(lambda d: d < np.Inf, [x for dgm in self.np_dgm() for x in dgm[:, 1]])
        return max(deaths) if len(deaths) else 0.
    def plot_dgm(self, axis):
        axis.cla()
        t = self.dgm_thresh()
        axis.plot([0, t], [0, t], c='black', alpha=0.5)
        for dgm in self.np_dgm():
            axis.scatter(dgm[:,0], dgm[:,1], s=5)
    def lfilt(self, f):
        return dio.Filtration(filter(f, self.F))

class DioRips(DioWrap):
    def __init__(self, pfun, data, dim, t):
        DioWrap.__init__(self, dio.fill_rips, pfun)
        self.build_filt(data, dim, t)
        self.data, self.dim, self.t = data, dim, t
    def threshold(self, t):
        return self.lfilt(lambda s: s.data <= t)
    def __getitem__(self, i):
        return self.F[i]
    def __setitem__(self, i, s):
        self.F[i] = s
    def __iter__(self):
        return self.F.__iter__()
    def __len__(self):
        return self.F.__len__()
    def __repr__(self):
        return self.F.__repr__()
    def __str__(self):
        return self.F.__str__()

class RipsCohomology(DioRips):
    def __init__(self, data, dim, t, prime):
        DioRips.__init__(self, dio.cohomology_persistence, data, dim, t)
        self.build_pers(prime, True)
        self.prime = prime
    def cocycle(self, pt):
        return self.H.cocycle(pt.data)
    def coords(self, pt):
        c, f = self.cocycle(pt), self.threshold((pt.death + pt.birth) / 2.)
        return np.array(dio.smooth(f, c, self.prime))
    def plot_coords(self, axis, v):
        color = cm.ScalarMappable(Normalize(v.min(), v.max()), cm.rainbow)
        axis.scatter(self.data[:,0], self.data[:,1], s=20, c=map(color.to_rgba, v))
