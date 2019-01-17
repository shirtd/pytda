from matplotlib.colors import Normalize
from scipy.sparse import csc_matrix
import matplotlib.cm as cm
import dionysus as dio
import numpy as np

def lfilt(f, F):
    return dio.Filtration(filter(f, F))

class DioWrap:
    def __init__(self, ffun, pfun, data):
        self.ffun, self.pfun, self.data = ffun, pfun, data
    def build_filt(self, *args, **kwargs):
        self.F = self.ffun(self.data, *args, **kwargs)
    def build_pers(self, *args, **kwargs):
        self.H = self.pfun(self.F, *args, **kwargs)
        self.D = dio.init_diagrams(self.H, self.F)

class DioFilt(DioWrap):
    def __init__(self, ffun, pfun, data):
        DioWrap.__init__(self, ffun, pfun, data)
    def __setitem__(self, i, s):
        self.F[i] = s
    def __getitem__(self, i): return self.F[i]
    def __contains__(self, s): return s in self.F
    def __iter__(self): return self.F.__iter__()
    def __len__(self): return self.F.__len__()
    def __repr__(self): return self.F.__repr__()
    def __str__(self): return self.F.__str__()
    def add(self, s): return self.F.add(s)
    # def append(self, s): return self.F.append(s)
    def index(self, s): return self.F.index(s)
    def rearrange(self, I): return self.F.rearrange(I)
    def sort(self): return self.F.sort()
    def lfilt(self, f): return lfilt(f, self.F)
    def np_filt(self, dim):
        return np.array([list(s) for s in self.F if s.dimension() == dim])
    def threshold(self, t):
        return self.lfilt(lambda s: s.data <= t)
    def restrict(self, V):
        return self.lfilt(lambda s: all(v in V for v in s))

class DioPersist(DioFilt):
    def __init__(self, ffun, pfun, data, prime):
        DioFilt.__init__(self, ffun, pfun, data)
        self.prime = prime
    def pair(self, pt): return self.H.pair(pt)
    def unpaired(self): return self.H.unpaired()
    def dgm_dim(self, dim):
        return sorted(self.D[dim], key=lambda p: p.death - p.birth, reverse=True)
    def get_point(self, dim, i=0):
        return self.dgm_dim(dim)[i]
    def np_dgm_dim(self, dim):
        return np.array([[p.birth, p.death] for p in self.dgm_dim(dim)])
    def np_dgm(self):
        return [self.np_dgm_dim(dim) for dim in range(len(self.D))]
    def dgm_thresh(self):
        l = [x for dgm in self.np_dgm() if len(dgm) for x in dgm[:, 1]]
        deaths = np.array(filter(lambda d: d < np.Inf, l))
        return deaths.max() if len(deaths) else 0.
    def np_cycle(self, c):
        return np.array([list(self.F[s.index]) for s in c])
    def plot_dgm(self, axis):
        axis.cla()
        t = self.dgm_thresh()
        axis.plot([0, t], [0, t], c='black', alpha=0.5)
        for dgm in self.np_dgm():
            if len(dgm):
                axis.scatter(dgm[:,0], dgm[:,1], s=5)

class DioHomology(DioPersist):
    def __init__(self, ffun, data, prime, Q, *args, **kwargs):
        DioPersist.__init__(self, ffun, dio.homology_persistence, data, prime)
        self.build_filt(*args, **kwargs)
        if len(Q):
            self.build_pers(self.prime, progress=True)
        else:
            self.Q, self.relative = Q, self.restrict(Q)
            self.build_pers(self.relative, self.prime, progress=True)
    def cycle(self, pt):
        return self.H[self.H.pair(pt.data)]

class DioCohomology(DioPersist):
    def __init__(self, ffun, data, prime, *args, **kwargs):
        DioPersist.__init__(self, ffun, dio.cohomology_persistence, data, prime)
        self.build_filt(*args, **kwargs)
        self.build_pers(self.prime, True)
    def cocycle(self, pt):
        return self.H.cocycle(pt.data)
    def coords(self, pt):
        c, f = self.cocycle(pt), self.threshold((pt.death + pt.birth) / 2.)
        return np.array(dio.smooth(f, c, self.prime))
    def plot_coords(self, axis, v):
        color = cm.ScalarMappable(Normalize(v.min(), v.max()), cm.rainbow)
        axis.scatter(self.data[:,0], self.data[:,1], s=20, c=map(color.to_rgba, v))
    def lift_simplex(self, f, s): # generalize to dim > 1
        i, bdy = f.index(s), s.boundary()
        return [(1. - 2. * (j % 2.), i, f.index(t)) for j, t in enumerate(bdy)]
    def lift_boundary(self, f, dim=1):
        data, row, col = zip(*np.concatenate([self.lift_simplex(f, s) for s in f if s.dimension() == dim]))
        row, col = map(lambda x: np.array(x, dtype=int), (row, col))
        k = max(max(row), max(col)) + 1
        return csc_matrix((np.array(data), (row, col)), shape=(k, k))
    def lift_cocycle(self, c, bdy):
        row_max, k = bdy.indices.max(), bdy.shape[0]
        z = filter(lambda x: x.index <= row_max, c)
        z_col = np.zeros(len(z), dtype=int)
        z_row, elem = zip(*map(lambda x: (x.index, x.element), z))
        z_data = [e if e < self.prime / 2. else e - self.prime for e in elem]
        return csc_matrix((z_data, (z_row, z_col)), shape=(k, 1)).toarray()
    def to_integer(self, f, c):
        bdy = self.lift_boundary(f)
        row_max = bdy.indices.max() # x.index <= row_max condition below projects the cocycle to the filtration
        z_bdy = self.lift_cocycle(c, bdy)
        # bdy = csc_matrix((np.array(data), (np.array(row), np.array(col))), shape=(dim, dim))
        F = dio.Filtration([s for i, s in enumerate(f) if i <= bdy.indices.max()])
        return F, bdy, z_bdy
    def get_lift(self, pt):
        c, f = self.cocycle(pt), self.threshold((pt.death + pt.birth) / 2.)
        return self.to_integer(f, c)

class RipsHomology(DioHomology):
    def __init__(self, data, dim, t, prime, Q=[]):
        DioHomology.__init__(self, dio.fill_rips, data, prime, Q, dim, t)
        self.dim, self.t = dim, t

class RipsCohomology(DioCohomology):
    def __init__(self, data, dim, t, prime):
        DioCohomology.__init__(self, dio.fill_rips, data, prime, dim, t)
        self.dim, self.t = dim, t

class StarHomology(DioHomology):
    def __init__(self, data, prime, Q=[], reverse=False):
        DioHomology.__init__(self, dio.fill_freudenthal, data, prime, Q, reverse)
        self.reverse = reverse

class StarCohomology(DioCohomology):
    def __init__(self, data, prime, reverse=False):
        DioCohomology.__init__(self, dio.fill_freudenthal, data, prime, reverse)
        self.reverse = reverse
