from ..hsn import *
from rips import *
from .. import *
from . import *

class HSNPlot(DioHomology, PersistencePlot):
    def __init__(self, fig, ax, net):
        PersistencePlot.__init__(self, fig, ax, 'homology', False)
        self.data, self.Q, self.F = net.data, net.Q, net.F
        self.relative, self.H, self.D = net.relative, net.H, net.D
        self.prime, self.dim, self.t = net.prime, net.dim, net.beta
        self.initialize(range(1, self.dim + 2))
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

class HSNInteract(HSN, InteractCmd):
    def __init__(self, X, dim, bound, alpha, beta, noise=0., prime=2, delta=0.02):
        fig, ax = get_axes(1, 3, figsize=(11, 4))
        classes = {'homology' : HomologyPlot, 'cohomology' : CohomologyPlot,
                    'circular' : CircularPlot, 'relative' : HSNPlot}
        HSN.__init__(self, X, dim, bound, alpha, beta, noise, fig, ax, delta, prime)
        InteractCmd.__init__(self, fig, ax, classes, self.F,
                            self.data, prime, dim, beta, False)
        # self.t = beta
        self.do_domain()
    def do_domain(self, arg=None): self.domain_plot()
    def do_size(self, arg=None): self.size_plot()
    def do_close(self, arg=None): self.close_plot()
    def do_net(self, arg=None): self.plot(self.ax)
    def do_tcc(self, arg=None):
        print('[ D \ B^%0.2f is%s covered' % (2 * self.alpha, '' if self.tcc()[0] else ' not'))
    def do_relative(self, arg=None): self.function('relative', self)
    def do_cohomology(self, arg=None): self.function('cohomology')
    def do_homology(self, arg=None): self.function('homology')
    def do_circular(self, arg=None): self.function('circular')
    def do_birth(self, arg):
        try:
            if self.fun != 'circular' and self.fun != 'relative':
                self.OBJ.birth = (arg in ['True', 'true'])
                self.OBJ.replot()
        except:
            print(' ! invalid arg %s' % arg)
    def do_thresh(self, arg):
        try:
            if self.fun != 'circular' and self.fun != 'relative':
                t = np.Inf if arg in ['inf', 'Inf'] else float(arg)
                self.OBJ.cycle_thresh = t
                self.OBJ.replot()
        except:
            print(' ! invalid thresh %s' % arg)
