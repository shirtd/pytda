from scipy.spatial import KDTree
from operator import mul
from ..persist import *
from .. import *

class PersistencePlot(DioPersist):
    def __init__(self, fig, ax, fun):
        self.fig, self.ax, self.fun = fig, ax, fun
    def initialize(self, dims):
        self.dims = dims
        self.plot_dgm(self.ax[2])
        self.dgms = self.sort_dgms_paired()
        dgms = map(self.to_np_dgm, self.dgms)
        self.kd = {d : KDTree(dgms[d]) for d in dims if len(dgms[d]) > 0}
        self.cache, self.last = {}, None
    def query(self, e):
        self.last = e
        p = np.array([e.xdata, e.ydata])
        if e.inaxes == self.ax[2]:
            ds = [(k, v.query(p)) for k, v in self.kd.iteritems()]
            dim, (d, idx) = min(ds, key=lambda (k, d): d[1])
            pt = self.get_point(dim, idx)
            self.plot_dgm(self.ax[2])
            self.ax[2].scatter(pt.birth, pt.death, s=20, zorder=2, c='black')
            return pt
        return None
    def plot_edges(self, axis, E, **kw):
        kw['c'] = kw['c'] if 'c' in kw else 'black'
        kw['alpha'] = kw['alpha'] if 'alpha' in kw else 0.3
        kw['zorder'] = kw['zorder'] if 'zorder' in kw else 1
        map(lambda e: axis.plot(e[:,0], e[:,1], **kw), E)
    def plot_data(self, axis, x=[], **kw):
        x = self.data if not len(x) else x
        kw['s'] = kw['s'] if 's' in kw else 10
        kw['zorder'] = kw['zorder'] if 'zorder' in kw else 2
        axis.scatter(self.data[:,0], self.data[:,1], **kw)

class HomologyPlot(SimplicialHomology, PersistencePlot):
    def __init__(self, fig, ax, F, data, dim, t, prime, Q=[]):
        PersistencePlot.__init__(self, fig, ax, 'homology')
        SimplicialHomology.__init__(self, F, data, prime, Q, dim=dim, t=t)
        self.initialize(range(1, dim + 1))
    def get_cycle(self, pt):
        return self.np_cycle(self.cycle(pt))
    def plot(self, key):
        if not key in self.cache:
            self.cache[key] = self.get_cycle(key)
        self.ax[1].cla()
        self.plot_data(self.ax[1])
        self.plot_edges(self.ax[1], self.data[self.cache[key]])

class CohomologyPlot(SimplicialCohomology, PersistencePlot):
    def __init__(self, fig, ax, F, data, dim, t, prime):
        PersistencePlot.__init__(self, fig, ax, 'cohomology')
        SimplicialCohomology.__init__(self, F, data, prime, dim=dim, t=t)
        self.initialize(range(1, dim + 1))
    def get_cocycle(self, pt):
        return self.np_cycle(self.cocycle(pt))
    def plot(self, key):
        if not key in self.cache:
            self.cache[key] = self.get_cocycle(key)
        self.ax[1].cla()
        self.plot_data(self.ax[1])
        self.plot_edges(self.ax[1], self.data[self.cache[key]])

class CircularPlot(SimplicialCohomology, PersistencePlot):
    def __init__(self, fig, ax, F, data, dim, t, prime):
        prime = prime if prime != 2 else 11
        PersistencePlot.__init__(self, fig, ax, 'circular')
        SimplicialCohomology.__init__(self, F, data, prime, dim=dim, t=t)
        self.initialize([1])
    def plot(self, key):
        if not key in self.cache:
            self.cache[key] = self.coords(key)
        self.ax[1].cla()
        self.plot_coords(self.ax[1], self.cache[key])

CLASS = {'homology' : HomologyPlot,
        'cohomology' : CohomologyPlot,
        'circular' : CircularPlot}

class RipsInteract:
    def __init__(self, data, dim, t):
        self.fig, self.ax = get_axes(1, 3, figsize=(11, 4))
        map(lambda x: x.axis('equal'), self.ax)
        self.data, self.dim, self.t = data, dim, t
        self.F = dio.fill_rips(self.data, self.dim, self.t)
        self.cache = {}
    def get_obj(self, fun, *args, **kwargs):
        if not fun in self.cache: # and self.cache[fun]
            self.cache[fun] = CLASS[fun](self.fig, self.ax, *args, **kwargs)
        return self.cache[fun]
    def anevent(self, fun, prime=2):
        self.OBJ = self.get_obj(fun, self.F, self.data, self.dim, self.t, prime)
        self.OBJ.plot_data(self.ax[0], c='black')
        def event(e):
            key = self.OBJ.query(e)
            if key: self.OBJ.plot(key)
        return event
    def addevent(self, *args, **kwargs):
        fun = self.anevent(*args, **kwargs)
        return self.fig.canvas.mpl_connect('button_release_event', fun)
