from .. import *
from . import *

class HomologyPlot(SimplicialHomology, PersistencePlot):
    def __init__(self, fig, ax, F, data, prime, dim, t, Q=[], birth=True):
        PersistencePlot.__init__(self, fig, ax, 'homology', birth)
        SimplicialHomology.__init__(self, F, data, prime, Q, dim=dim, t=t)
        self.initialize(range(1, dim + 1))
    def get_cycle(self, pt):
        return self.np_cycle(self.cycle(pt))
    def plot(self, key):
        print('[ cycle of point '+ str(key))
        if not key in self.cache:
            self.cache[key] = self.cycle(key)
        if self.cache[key] != None:
            self.clear_ax(self.ax[1])
            self.plot_data(self.ax[1], zorder=2)
            # c = self.thresh_cycle(key)
            c = [self.F[s.index] for s in self.cache[key]]
            e, t = np.array(map(list, c)), max(s.data for s in c)
            if len(c): self.plot_edges(self.ax[1], self.data[e], alpha=1, linewidth=2, c='red', zorder=1)
            # C = self.lfilt(lambda s: s.dimension() == 1 and s.data <= t and not s in c)
            # E = np.array(map(list, C))
            # self.plot_edges(self.ax[1], self.data[E], False, c='black', alpha=0.25, linewidth=0.5, zorder=0)

class CohomologyPlot(SimplicialCohomology, PersistencePlot):
    def __init__(self, fig, ax, F, data, prime, dim, t, birth=True):
        PersistencePlot.__init__(self, fig, ax, 'cohomology', birth)
        SimplicialCohomology.__init__(self, F, data, prime, dim=dim, t=t)
        self.initialize(range(1, dim + 1))
    def plot(self, key):
        print('[ cocycle of point '+ str(key))
        if not key in self.cache:
            self.cache[key] = self.cocycle(key)
        self.clear_ax(self.ax[1])
        self.plot_data(self.ax[1], zorder=2)
        # c = self.thresh_cycle(key)
        c = [self.F[s.index] for s in self.cache[key]]
        e, t = np.array(map(list, c)), max(s.data for s in c)
        if len(c): self.plot_edges(self.ax[1], self.data[e], alpha=1, linewidth=1, c='red', zorder=1)
        # C = self.lfilt(lambda s: s.dimension() == 1 and s.data <= t and not s in c)
        # E = np.array(map(list, C))
        # self.plot_edges(self.ax[1], self.data[E], False, c='black', alpha=0.25, linewidth=0.5, zorder=0)

class CircularPlot(SimplicialCohomology, PersistencePlot):
    def __init__(self, fig, ax, F, data, prime, dim, t, birth=False):
        prime = prime if prime != 2 else 11
        PersistencePlot.__init__(self, fig, ax, 'circular', False)
        SimplicialCohomology.__init__(self, F, data, prime, dim=dim, t=t)
        self.initialize([1])
    def plot(self, key):
        print('[ circular coordinates from point '+ str(key))
        if not key in self.cache:
            self.cache[key] = self.coords(key)
        self.clear_ax(self.ax[1])
        self.plot_coords(self.ax[1], self.cache[key])

class RipsInteract(InteractCmd):
    def __init__(self, data, dim, t, prime=2, fun='circular'):
        fig, ax = get_axes(1, 3, figsize=(11, 4))
        classes = {'homology' : HomologyPlot,
                    'cohomology' : CohomologyPlot,
                    'circular' : CircularPlot}
        F = dio.fill_rips(data, dim, t)
        kw = {'F' : F, 'data' : data, 'prime' : prime, 'dim' : dim, 't' : t}
        InteractCmd.__init__(self, fig, ax, classes, **kw)
        self.dim, self.t, self.prime = dim, t, prime
        self.do_fun(fun)
    def do_birth(self, arg):
        try:
            if self.fun != 'circular':
                self.OBJ.birth = (arg in ['True', 'true'])
                self.OBJ.replot()
        except:
            print(' ! invalid arg %s' % arg)
    def do_thresh(self, arg):
        try:
            if self.fun != 'circular':
                t = np.Inf if arg in ['inf', 'Inf'] else float(arg)
                self.OBJ.cycle_thresh = t
                self.OBJ.replot()
        except:
            print(' ! invalid thresh %s' % arg)
    def do_cohomology(self, arg):
        self.do_fun('cohomology')
    def do_homology(self, arg):
        self.do_fun('homology')
    def do_circular(self, arg):
        self.do_fun('circular')
