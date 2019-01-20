from collections import defaultdict
from itertools import combinations
from ..persist import *
from network import *
from .. import *

class GaussianNetwork:
    def __init__(self, X, alpha, beta, bound, ax=[]):
        self.X, self.ax = X, ax
        self.alpha, self.beta = alpha, beta
        self.extent = (-4.5, len(X) + 3.5, -4.5, len(X) + 3.5)
        self.bound = self.find_boundary(bound, len(ax) > 0)
    ''''''''''''''''''''''''''''''''''''''''''''''''
    '''  generate domain satisfying assumptions  '''
    def find_boundary(self, bound, plot=False, delta=0.02):
        while True:
            print('[ bound = %0.2f' % bound)
            self.get_domain(bound, plot)
            if self.test_size(plot):
                if self.test_close(plot):
                    break
            bound += delta
            delete_line()
        return bound
    ''''''''''''''''''''''''''''''''''''
    '''  generate gaussian network   '''
    def get_domain(self, bound, plot=False):
        _n, n = len(self.X), len(self.X) + 8
        G0_idx, G_idx = grid_idx(_n), grid_idx(range(-4, _n + 4))
        G, G0 = G_idx.reshape(-1, 2), G0_idx.reshape(-1, 2)
        D0_idx = np.vstack(filter(lambda (i, j): self.X[i, j] <= bound, G0))
        self.M = distance_grid(D0_idx, G_idx)
        C0 = find_contours_ext(self.M.T, self.alpha + 1e-1)
        T = spatial.KDTree(G)
        Y = [l for l, (i, j) in enumerate(G + 4) if self.M.T[i, j] <= self.alpha]
        L = np.unique(np.concatenate(T.query_ball_point(np.vstack(C0), self.alpha)))
        I = np.intersect1d(L, Y)
        J = [i for i in Y if not i in I]
        K = [i for i in range(len(G)) if not (i in I or i in J)]
        # I = [l for l, (i, j) in enumerate(G + 4) if l in I and self.M.T[i, j] <= self.alpha]
        # J = [l for l, (i, j) in enumerate(G + 4) if not l in L and self.M.T[i, j] < self.alpha]
        # K = [l for l, (i, j) in enumerate(G + 4) if not l in J and self.M.T[i, j] > self.alpha]
        # I = [i for i in range(len(G)) if not (i in J or i in K)]
        # # K = [i for i in range(len(G)) if not (i in I or i in J)]
        self.INT, self.B, self.C = map(lambda x: np.array(x, dtype=float), (G[J], G[I], G[K]))
        self.MC = distance_grid(self.C, G_idx)
        self.MD = distance_grid(np.vstack((self.B, self.INT)), G_idx)
        if plot:
            self.clear_ax()
            self.plot_X(self.ax[0])
            self.plot_M(self.ax[1])
            plot_contours(self.ax[1], C0, c='black')
            self.plot_all(self.ax[2])
            plot_contours(self.ax[2], C0, c='black')
            plt.show(False)
            wait(' : domain')
    ''''''''''''''''''''
    ''' assumption 1 '''
    def test_size(self, plot=False):
        C0 = find_contours_ext(self.MC.T, self.alpha + self.beta)
        C1 = find_contours_ext(self.MC.T, 2 * self.alpha)
        poly0, poly1 = map(poly_contour, (C0, C1))
        if plot:
            self.clear_ax()
            self.plot_all(self.ax[0])
            self.plot_MC(self.ax[1])
            plot_contours(self.ax[1], C0, c='red')
            self.plot_MC(self.ax[2])
            plot_contours(self.ax[2], C1, c='blue')
            plt.show(False)
            wait(' : test_size  (%d -> %d)' % (len(poly0), len(poly1)))
        return (len(poly0) + len(poly1) and len(poly0) >= len(poly1)
            and all(any(p.within(q) for p in poly0) for q in poly1))
    ''''''''''''''''''''
    ''' assumption 2 '''
    def test_close(self, plot=False):
        C0 = find_contours_ext(self.MC.T, 2 * self.alpha)
        C1 = find_contours_ext(self.MD.T, 2 * self.alpha)
        poly0, poly1 = map(poly_contour, (C0, C1))
        if plot:
            self.clear_ax()
            self.plot_all(self.ax[0])
            self.plot_MC(self.ax[1])
            plot_contours(self.ax[1], C0, c='red')
            self.plot_MD(self.ax[2])
            plot_contours(self.ax[2], C1, c='blue')
            plt.show(False)
            wait(' : test_close (%d -> %d)' % (len(poly0), len(poly1)))
        return len(poly0) <= len(poly1) and all(any(p.within(q) for q in poly1) for p in poly0)
    ''''''''''''''''''''''''''''''''''''
    ''' plot network and assumptions '''
    def clear_ax(self): map(lambda x: x.cla(), self.ax)
    def plot_X(self, axis): axis.imshow(self.X.T, origin='lower')
    def plot_M(self, axis): axis.imshow(self.M, origin='lower', extent=self.extent)
    def plot_MC(self, axis): axis.imshow(self.MC, origin='lower', extent=self.extent)
    def plot_MD(self, axis): axis.imshow(self.MD, origin='lower', extent=self.extent)
    def plot_domain(self, axis, **kw): axis.scatter(self.INT[:,0], self.INT[:,1], **kw)
    def plot_boundary(self, axis, **kw): axis.scatter(self.B[:,0], self.B[:,1], **kw)
    def plot_complement(self, axis, **kw): axis.scatter(self.C[:,0], self.C[:,1], **kw)
    def plot_all(self, axis):
        self.plot_domain(axis, s=3, c='blue', zorder=1)
        self.plot_boundary(axis, s=5, c='red', zorder=0)
        self.plot_complement(axis, s=3, c='black', zorder=2)

class HSN(RipsHomology, GaussianNetwork):
    def __init__(self, X, dim, bound, alpha, beta, ax=[], delta=0.02, prime=2):
        GaussianNetwork.__init__(self, X, alpha, beta, bound, ax)
        data, Q = np.vstack((self.B, self.INT)), range(len(self.B))
        print('[ %d point interior, %d point boundary' % (len(self.INT), len(self.B)))
        RipsHomology.__init__(self, data, dim + 1, beta, prime, Q)
        self.dim = self.dim - 1
        if len(self.ax):
            self.plot(self.ax)
            plt.show(False)
    def in_int(self, s):
        return all(not v in self.Q for v in s)
    def in_bdy(self, s):
        return all(v in self.Q for v in s)
    def get_cycle(self, pt):
        c = self.cycle(pt)
        return self.np_cycle(c) if c != None else np.empty((0,2))
    def get_cycles(self):
        return [self.get_cycle(pt) for pt in pts if self.is_paired(pt)]
    def restrict_domain(self, t):
        TB = spatial.KDTree(self.B)
        fun = lambda i: TB.query(self.data[i])[0] > t
        return filter(fun, range(len(self.data)))
    def components(self, t):
        print('[ finding connected components of D \ B^%0.4f' % t)
        # F = self.restrict(self.restrict_domain(t))
        F = dio.fill_rips(self.INT, 2, self.alpha)
        H = dio.homology_persistence(F, 2, 'clearing', True)
        D = map(self.sort_dgm, dio.init_diagrams(H, F))
        return filter(lambda p: H.pair(p.data) == H.unpaired, D[0])
    def get_unpaired_points(self, dim, t=np.Inf):
        return filter(lambda p: not self.is_paired(p) and p.birth <= t, self.D[dim])
    # def get_unpaired(self, dim, t=np.Inf):
    #     # return map(lambda p: (p.birth, self.F[p.data]), self.get_unpaired_points(dim, t))
    #     return map(lambda p: self.F[p.data], self.get_unpaired_points(dim, t))
    # # def birth_dict(self, up):
    # #     bdict = defaultdict(set)
    # #     for b, s in up:
    # #         bdict[b].add(s)
    # #     return dict(bdict)
    # def birth_dict(self, up):
    #     bdict = defaultdict(set)
    #     for s in up:
    #         bdict[s.data].add(s)
    #     return dict(bdict)
    # def betti_birth(self, dim, t=np.Inf):
    #     # return self.birth_dict(self.get_unpaired(dim, t))
    #     return self.get_unpaired(dim, t)
    def tcc(self):
        # HRD = self.betti_birth(self.dim, self.alpha)
        HRD = self.get_unpaired_points(self.dim, self.alpha)
        H0 = self.components(self.alpha)
        return len(H0) == len(HRD), HRD, H0
    def plot_network(self, axis):
        axis.scatter(self.INT[:,0], self.INT[:,1], s=5, c='blue', zorder=3)
        axis.scatter(self.B[:,0], self.B[:,1], s=10, c='red', zorder=4)
    def plot(self, axes, clear=True):
        self.clear_ax()
        F0 = self.restrict(self.restrict_domain(self.alpha))
        axes[0].scatter(self.B[:,0], self.B[:,1], s=3, c='red')
        self.plot_simplices(axes[0], lfilt(lambda s: s.data <= self.alpha, F0))
        F1 = self.restrict(self.restrict_domain(2 * self.alpha))
        axes[1].scatter(self.B[:,0], self.B[:,1], s=3, c='red')
        self.plot_simplices(axes[1], lfilt(lambda s: s.data <= self.alpha, F1))
        F2 = self.restrict(self.restrict_domain(self.beta))
        axes[2].scatter(self.B[:,0], self.B[:,1], s=3, c='red')
        self.plot_simplices(axes[2], lfilt(lambda s: s.data <= self.alpha, F2))
    def net_dict(self):
        return {'bound' : self.bound, 'alpha' : self.alpha, 'beta' : self.beta,
                'X' : self.X, 'INT' : self.INT, 'B' : self.B, 'C' : self.C,
                'data' : self.data, 'MC' : self.MC, 'MD' : self.MD, 'M' : self.M,
                'dgm' : self.np_dgm(), 'F' : self.fdict()}

        # self.plot_network(axes[0])
        # self.plot_cover(axes[1], self.alpha, color='blue')
        # self.plot_cover(axes[1], self.alpha, self.Q, color='red')
        # self.plot_simplices(axes[1], self.lfilt(lambda s: s.data <= self.alpha))
        # F0 = self.restrict(self.restrict_domain(self.alpha))
