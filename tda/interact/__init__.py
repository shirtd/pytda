from cmd import Cmd
from ..plot import *
from .. import *
import argparse

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
        self.clear_ax(self.ax[1])
        self.plot_data(self.ax[1], zorder=2)
        c = self.thresh_cycle(key)
        e, t = np.array(map(list, c)), max(s.data for s in c)
        if len(c): self.plot_edges(self.ax[1], self.data[e], alpha=1, linewidth=2, c='red', zorder=1)
        C = self.lfilt(lambda s: s.dimension() == 1 and s.data <= t and not s in c)
        E = np.array(map(list, C))
        self.plot_edges(self.ax[1], self.data[E], False, c='black', alpha=0.25, linewidth=0.5, zorder=0)

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
        c = self.thresh_cycle(key)
        e, t = np.array(map(list, c)), max(s.data for s in c)
        if len(c): self.plot_edges(self.ax[1], self.data[e], alpha=1, linewidth=1, c='red', zorder=1)
        C = self.lfilt(lambda s: s.dimension() == 1 and s.data <= t and not s in c)
        E = np.array(map(list, C))
        self.plot_edges(self.ax[1], self.data[E], False, c='black', alpha=0.25, linewidth=0.5, zorder=0)

class CircularPlot(SimplicialCohomology, PersistencePlot):
    def __init__(self, fig, ax, F, data, prime, dim, t):
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

class RipsInteract(Interact):
    def __init__(self, data, dim, t, prime=2):
        fig, ax = get_axes(1, 3, figsize=(11, 4))
        map(lambda x: x.axis('equal'), ax)
        map(lambda x: x.axis('off'), ax)
        classes = {'homology' : HomologyPlot,
                    'cohomology' : CohomologyPlot,
                    'circular' : CircularPlot}
        F = dio.fill_rips(data, dim, t)
        Interact.__init__(self, fig, ax, classes, F, data, prime, dim, t)
        self.dim, self.t, self.prime = dim, t, prime

class RipsCommand(RipsInteract, Cmd):
    def __init__(self, data, dim, t, prime=2, fun='cohomology'):
        RipsInteract.__init__(self, data, dim, t, prime)
        self.last, self.cid = None, None
        self.do_fun(fun)
        Cmd.__init__(self)
        self.prompt = '> '
    def do_fun(self, fun):
        self.fun = fun
        if self.cid != None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.OBJ.clear()
            self.last = self.OBJ.last
        self.cid = self.addevent(fun)
        if self.last != None:
            self.OBJ.last = self.last
            key = self.OBJ.query(self.last)
            if key: self.OBJ.plot(key)
        plt.show(False)
    def do_cohomology(self, args): self.do_fun('cohomology')
    def do_homology(self, args): self.do_fun('homology')
    def do_circular(self, args): self.do_fun('circular')
    def do_birth(self, args):
        if self.fun != 'circular':
            self.OBJ.birth = (args in ['True', 'true'])
            self.OBJ.replot()
    def do_thresh(self, args):
        if self.fun != 'circular':
            try:
                t = np.Inf if args in ['inf', 'Inf'] else float(args)
                self.OBJ.cycle_thresh = t
                self.OBJ.replot()
            except:
                print(' ! invalid threshold')
    def do_prime(self, args):
        try:
            self.prime, self.cache = int(args), {}
            self.do_fun(self.fun)
        except:
            print(' ! invalid prime')
    def do_save(self, args):
        try:
            i, fname = args.split()
            save_axis(self.fig, self.ax[int(i)], fname)
        except:
            print(' ! invalid argument %s' % args)
    def do_EOF(self, line):
        return True
    def do_exit(self, args):
        raise SystemExit
