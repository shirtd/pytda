from numpy import asarray, eye, outer, inner, dot, vstack
from numpy.random import seed, rand
from itertools import combinations
from scipy.sparse.linalg import cg
from pydec import d, delta, star
from ..plot import plot_edges
from numpy.linalg import norm
import numpy.linalg as la
from ..persist import *
import dionysus as dio
from pydec import *
import numpy as np

class RipsDEC(DioCohomology, dec.simplicial_complex):
    def __init__(self, H, pt):
        self.H, self.pt, self.dim = H, pt, H.dim
        F, self.bdy, self.z_bdy = H.get_lift(self.pt)
        print('[ %d simplex %dD restriction (from %d)' % (len(F), self.dim, len(H)))
        DioCohomology.__init__(self, lambda x, y: y, H.data, H.prime, F)
        self.V = self.get_vertices(self.dim)
        dec.simplicial_complex.__init__(self, (self.data, self.V))
    def __getitem__(self, i): return super(dec.simplicial_complex, self).__getitem__(i)
    def __iter__(self): return super(dec.simplicial_complex, self).__iter__()
    def append(self, l): return super(dec.simplicial_complex, self).append(l)
    def get_simplices(self, dim, t=np.Inf, eq=True):
        op = lambda x, y: x <= y if eq else x < y
        return [s for s in self.F if s.dimension() == dim and op(s.data, t)]
    def get_vertices(self, dim, t=np.Inf, eq=True):
        return np.array([[v for v in s] for s in self.get_simplices(dim, t, eq)])
    def fix_chain(self, z_bdy):
        K, E = self.complex(), self.get_vertices(1)
        c, L, S = self.get_cochain(1), list(map(list, K[1])), self.get_simplices(1)
        x = [(i, e) for i, e in enumerate(z_bdy.flatten()) if self.F[i].dimension() == 1]
        if len(K[1]) != len(E):
            diff = np.array(map(list, set(map(tuple, E)).difference(set(map(tuple, K[1])))))
            miss = np.array([np.where(np.all(E == e, axis=1))[0][0] for e in diff])
            for i in sorted(miss, reverse=True): del x[i]
        I, V = zip(*x)
        emap = {self.F.index(s) : L.index([v for v in s]) for i, s in enumerate(S) if not i in miss}
        c.v[[emap[i] for i in I]] = V
        return c
    def build_cochain(pt):
        C, c = self.cocycle(pt), self.get_cochain(1)
        I, V = zip(*[(e.index, e.element) for e in C])
        L = list(map(list, self.complex()[1]))
        S = self.get_simplices(1, pt.death)
        emap = {self.index(s) : L.index([v for v in s]) for s in S}
        c.v[[emap[i] for i in I]] = V
        return c
    def hodge(self):
        c = self.fix_chain(self.z_bdy)
        return self.hodge_decomposition(c)
    def hodge_decomposition(self, omega):
        print('[ constructing hodge decomposition')
        sc, p = omega.complex, omega.k
        alpha = sc.get_cochain(p - 1)
        beta  = sc.get_cochain(p + 1)
        print(' | solving for alpha')
        A = delta(d(sc.get_cochain_basis(p - 1))).v
        b = delta(omega).v
        alpha.v = cg( A, b, tol=1e-8 )[0]
        print(' | solving for beta')
        A = d(delta(sc.get_cochain_basis(p + 1))).v
        b = d(omega).v
        beta.v = cg( A, b, tol=1e-8 )[0]
        h = omega - d(alpha) - delta(beta) # Solve for h
        return alpha, beta, h
    def plot_edges(self, axis, h):
        K = self.complex()
        plot_edges(axis, self.data[K[1]], h.v, 0.02, alpha=0.5, zorder=0)
    def plot_vertices(self, axis, alpha):
        v = self.complex()[0].flatten()
        axis.scatter(self.data[v,0], self.data[v,1], c=alpha.v, cmap='rainbow', zorder=1)
    def plot_hodge(self, axis):
        alpha, beta, h = self.hodge()
        self.plot_vertices(axis, alpha)
        self.plot_edges(axis, h)
        return alpha, beta, h
