from collections import defaultdict
from itertools import combinations
from ..persist import *
from network import *
from ..plot import *
from .. import *

PAD = 8

class GaussianNetwork:
    def __init__(self, X, alpha, beta, bound, noise, fig=None, ax=[]):
        self.X, self.ax = X, ax
        if len(ax):
            self.fig = fig
        self.alpha, self.beta, self.noise = alpha, beta, noise
        self.extent = (-1.*(PAD + 0.5), len(X) + PAD-0.5,
                        -1.*(PAD + 0.5), len(X) + PAD-0.5)
        self.contour, self.poly = {}, {}
        self.bound = self.find_boundary(bound, len(ax) > 0)
    def add_noise(self, x):
        x = np.array(x, dtype=float)
        y = self.noise * np.random.rand(len(x), 2)
        return x + y - np.array([self.noise, self.noise]) / 2.
    ''''''''''''''''''''''''''''''''''''''''''''''''
    '''  generate domain satisfying assumptions  '''
    def find_boundary(self, bound, plot=False, delta=0.02):
        bound -= delta
        while bound < 1:
            bound += delta
            print('[ bound = %0.2f' % bound)
            self.get_domain(bound, plot)
            if self.test_size(plot) and self.test_close(plot): break
            delete_line()
        return bound
    ''''''''''''''''''''''''''''''''''''
    '''  generate gaussian network   '''
    def get_domain(self, bound, plot=False):
        _n, n = len(self.X), len(self.X) + 2 * PAD
        G0_idx, G_idx = grid_idx(_n), grid_idx(range(-PAD, _n + PAD))
        G, G0 = G_idx.reshape(-1, 2), G0_idx.reshape(-1, 2)
        _G = self.add_noise(G)
        D0_idx = np.vstack(filter(lambda (i, j): self.X[i, j] <= bound, G0))
        self.M = distance_grid(D0_idx, G_idx)
        B0 = find_contours_ext(self.M.T, self.noise, PAD)
        self.contour['B'] = B0
        T = spatial.KDTree(_G)
        Y = [l for l, (i, j) in enumerate(G + PAD) if self.M.T[i, j] <= self.noise]
        L = np.unique(np.concatenate(T.query_ball_point(np.vstack(B0), self.alpha)))
        I = np.intersect1d(L, Y)
        J = [i for i in Y if not i in I]
        K = [i for i in range(len(G)) if not (i in I or i in J)]
        self.INT, self.B, self.C = _G[J], _G[I], _G[K]
        self.MC = distance_grid(self.C, G_idx)
        self.MD = distance_grid(np.vstack((self.B, self.INT)), G_idx)
    ''''''''''''''''''''
    ''' assumption 1 '''
    def test_size(self, plot=False):
        C0 = find_contours_ext(self.MC.T, self.alpha + self.beta, PAD)
        C1 = find_contours_ext(self.MC.T, 2 * self.alpha, PAD)
        poly0, poly1 = map(poly_contour, (C0, C1))
        self.poly['size'] = (poly0, poly1)
        self.contour['size'] = (C0, C1)
        return (len(poly0) + len(poly1) and len(poly0) >= len(poly1)
            and all(any(p.within(q) for p in poly0) for q in poly1))
    ''''''''''''''''''''
    ''' assumption 2 '''
    def test_close(self, plot=False):
        C0 = find_contours_ext(self.MC.T, 2 * self.alpha, PAD)
        C1 = find_contours_ext(self.MD.T, 2 * self.alpha, PAD)
        poly0, poly1 = map(poly_contour, (C0, C1))
        self.poly['close'] = (poly0, poly1)
        self.contour['close'] = (C0, C1)
        return len(poly0) <= len(poly1) and all(any(p.within(q) for q in poly1) for p in poly0)
    ''''''''''''''''''''''''''''''''''''
    ''' plot network and assumptions '''
    def clear_ax(self):
        map(lambda x: x.cla(), self.ax)
        map(lambda x: x.axis('equal'), self.ax)
        map(lambda x: x.axis('off'), self.ax)
    def plot_X(self, axis): axis.imshow(self.X.T, origin='lower', interpolation='bilinear')
    def plot_M(self, axis): axis.imshow(self.M, origin='lower', extent=self.extent, interpolation='bilinear')
    def plot_MC(self, axis): axis.imshow(self.MC, origin='lower', extent=self.extent, interpolation='bilinear')
    def plot_MD(self, axis): axis.imshow(self.MD, origin='lower', extent=self.extent, interpolation='bilinear')
    def domain_plot(self, shadow=True):
        self.clear_ax()
        self.plot_X(self.ax[0])
        self.plot_M(self.ax[1])
        plot_contours(self.ax[1], self.contour['B'], shadow, c='black')
        self.plot_all(self.ax[2], shadow)
        plot_contours(self.ax[2], self.contour['B'], shadow, c='black')
        # plt.show(False)
        # for i, ax in enumerate(self.ax):
        #     save_axis(self.fig, ax, 'tex2/figures/hsn_domain_%d.pdf' % i)
    def size_plot(self, shadow=True):
        self.clear_ax()
        self.plot_all(self.ax[0], shadow)
        self.plot_MC(self.ax[1])
        self.plot_MC(self.ax[2])
        p0, p1 = self.poly['size']
        plot_poly(self.ax[1], p0, shadow, c='red', alpha=0.3)
        plot_poly(self.ax[2], p1, shadow, c='blue', alpha=0.3)
        # plt.show(False)
        # for i, ax in enumerate(self.ax):
        #     save_axis(self.fig, ax, 'tex2/figures/hsn_size_%d.pdf' % i)
    def close_plot(self, shadow=True):
        self.clear_ax()
        self.plot_all(self.ax[0], shadow)
        self.plot_MC(self.ax[1])
        self.plot_MD(self.ax[2])
        p0, p1 = self.poly['close']
        plot_poly(self.ax[1], p0, shadow, c='red', alpha=0.3)
        plot_poly(self.ax[2], p1, shadow, c='blue', alpha=0.15)
        # plt.show(False)
        # for i, ax in enumerate(self.ax):
        #     save_axis(self.fig, ax, 'tex2/figures/hsn_close_%d.pdf' % i)
    def plot_domain(self, axis, shadow=True, **kw):
        if 'color' in kw:
            kw['c'] = kw['color']
            del kw['color']
        kw['c'] = 'blue' if not 'c' in kw else kw['c']
        kw['zorder'] = 1 if not 'zorder' in kw else kw['zorder']
        kw['markersize'] = 1 if not 'markersize' in kw else kw['markersize']
        if shadow:
            kw['path_effects'] = [pfx.withSimplePatchShadow()]
        axis.plot(self.INT[:,0], self.INT[:,1], 'o', **kw)
    def plot_boundary(self, axis, shadow=True, **kw):
        if 'color' in kw:
            kw['c'] = kw['color']
            del kw['color']
        kw['c'] = 'red' if not 'c' in kw else kw['c']
        kw['zorder'] = 0 if not 'zorder' in kw else kw['zorder']
        kw['markersize'] = 1.5 if not 'markersize' in kw else kw['markersize']
        if shadow:
            kw['path_effects'] = [pfx.withSimplePatchShadow()]
        axis.plot(self.B[:,0], self.B[:,1], 'o', **kw)
    def plot_complement(self, axis, shadow=False, **kw):
        if 'color' in kw:
            kw['c'] = kw['color']
            del kw['color']
        kw['c'] = 'black' if not 'c' in kw else kw['c']
        kw['zorder'] = 2 if not 'zorder' in kw else kw['zorder']
        kw['markersize'] = 0.5 if 'markersize' not in kw else kw['markersize']
        if shadow:
            kw['path_effects'] = [pfx.withSimplePatchShadow()]
        axis.plot(self.C[:,0], self.C[:,1], 'o', **kw)
    def plot_all(self, axis, shadow=True):
        self.plot_domain(axis, shadow, c='blue', zorder=1)
        self.plot_boundary(axis, shadow, c='red', zorder=0)

class HSN(RipsHomology, GaussianNetwork):
    def __init__(self, X, dim, bound, alpha, beta, noise=0., fig=None, ax=[], delta=0.02, prime=2):
        GaussianNetwork.__init__(self, X, alpha, beta, bound, noise, fig, ax)
        data, Q = np.vstack((self.B, self.INT)), range(len(self.B))
        print('[ %d point interior, %d point boundary' % (len(self.INT), len(self.B)))
        RipsHomology.__init__(self, data, dim + 1, beta, prime, Q)
        self.dim = self.dim - 1
        # if len(self.ax):
        #     self.plot(self.ax)
        #     # plt.show(False)
        #     # for i, ax in enumerate(self.ax):
        #     #     save_axis(self.fig, ax, 'tex2/figures/hsn_net_%d.pdf' % i)
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
        F = dio.fill_rips(self.INT, 2, self.alpha)
        H = dio.homology_persistence(F, 2, 'clearing', True)
        D = map(self.sort_dgm, dio.init_diagrams(H, F))
        return filter(lambda p: H.pair(p.data) == H.unpaired, D[0])
    def get_unpaired_points(self, dim, t=np.Inf):
        return filter(lambda p: not self.is_paired(p) and p.birth <= t, self.D[dim])
    def tcc(self):
        HRD = self.get_unpaired_points(self.dim, self.alpha)
        H0 = self.components(self.alpha)
        return len(H0) == len(HRD), HRD, H0
    def plot_network(self, axis):
        axis.scatter(self.INT[:,0], self.INT[:,1], s=5, c='blue', zorder=3)
        axis.scatter(self.B[:,0], self.B[:,1], s=10, c='red', zorder=4)
    def plot(self, axes, shadow=True, clear=True):
        self.clear_ax()
        A = self.lfilt(lambda s: s.data <= self.alpha)
        Q = lfilt(lambda s: all(v in self.Q for v in s), A)
        INT = lfilt(lambda s: all(v in range(max(self.Q), len(self.data)) for v in s), A)
        self.plot_simplices(axes[0], A, shadow, c='black')
        xs, ys = axes[0].get_ylim(), axes[0].get_ylim()
        self.plot_simplices(axes[1], Q, shadow, c='red')
        self.plot_simplices(axes[2], INT, shadow, c='blue')
        # F1 = self.restrict(self.restrict_domain(2 * self.alpha))
        # self.plot_simplices(axes[2], lfilt(lambda s: s.data <= self.alpha, F1))
        map(lambda x: x.set_xlim(*xs), axes)
        map(lambda x: x.set_ylim(*ys), axes)
    def net_dict(self):
        return {'bound' : self.bound, 'alpha' : self.alpha, 'beta' : self.beta,
                'X' : self.X, 'INT' : self.INT, 'B' : self.B, 'C' : self.C,
                'data' : self.data, 'MC' : self.MC, 'MD' : self.MD, 'M' : self.M,
                'dgm' : self.np_dgm()} #, 'F' : self.fdict()}

''''''''''''''''''
''' DEPRECATED '''
''''''''''''''''''

# def plot(self, axes, clear=True):
#     self.clear_ax()
#     F0 = self.restrict(self.restrict_domain(self.alpha))
#     axes[0].scatter(self.B[:,0], self.B[:,1], s=3, c='red')
#     self.plot_simplices(axes[0], lfilt(lambda s: s.data <= self.alpha, F0))
#     F1 = self.restrict(self.restrict_domain(2 * self.alpha))
#     axes[1].scatter(self.B[:,0], self.B[:,1], s=3, c='red')
#     self.plot_simplices(axes[1], lfilt(lambda s: s.data <= self.alpha, F1))
#     F2 = self.restrict(self.restrict_domain(self.beta))
#     axes[2].scatter(self.B[:,0], self.B[:,1], s=3, c='red')
#     self.plot_simplices(axes[2], lfilt(lambda s: s.data <= self.alpha, F2))
