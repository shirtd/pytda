from ..plot import *
from .. import *
import argparse

class HomologyPlot(SimplicialHomology, PersistencePlot):
    def __init__(self, fig, ax, F, data, dim, t, prime, Q=[]):
        PersistencePlot.__init__(self, fig, ax, 'homology')
        SimplicialHomology.__init__(self, F, data, prime, Q, dim=dim, t=t)
        self.initialize(range(1, dim + 1))
    def get_cycle(self, pt):
        return self.np_cycle(self.cycle(pt))
    def plot(self, key):
        if not key in self.cache:
            # print(' | saving cycle from point '+ str(key))
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
            # print(' | getting cocycle from point '+ str(key))
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
            # print(' | calculating circular coordinates from point '+ str(key))
            self.cache[key] = self.coords(key)
        self.ax[1].cla()
        self.plot_coords(self.ax[1], self.cache[key])

class RipsInteract(Interact):
    def __init__(self, data, dim, t, prime=2):
        fig, ax = get_axes(1, 3, figsize=(11, 4))
        map(lambda x: x.axis('equal'), ax)
        classes = {'homology' : HomologyPlot,
                    'cohomology' : CohomologyPlot,
                    'circular' : CircularPlot}
        F = dio.fill_rips(data, dim, t)
        Interact.__init__(self, fig, ax, classes, F, data, dim, t, prime)
        self.dim, self.t, self.prime = dim, t, prime

parser = argparse.ArgumentParser(description='interactive cycles, cocycles, and circular coordinates.')
parser.add_argument('-f','--fun', default='circular', help='persistence function. default: homology.')
# parser.add_argument('dset', default='mnist', nargs='?', help='data set. default: mnist.')
# parser.add_argument('-c','--c', default=0, type=int, help='image class. default: 0.')
# parser.add_argument('-i','--i', default=0, type=int, help='image index. default: 0.')
# parser.add_argument('-p','--prime', default=11, type=int, help='field coefficient. default: 2 (11 if fun = circular).')

CMDS = {'fun' : lambda v: v in ['homology', 'cohomology', 'circular']} #, 'prime' : lambda v: v.isdigit()}
FDICT = {'fun' : lambda v: v} #, 'prime' : lambda v: int(v)}

def process_cmd(args, input):
    cmds = input.split(',') if ',' in input else [input]
    for cmd in cmds:
        k, v = cmd.split(' ')
        if k in CMDS and CMDS[k](v):
            args.__setattr__(k, FDICT[k](v))
        elif k in FDICT:
            print('! invalid argument %s for command %s' % (v, k))
        else:
            print('! unknown command %s' % k)
    return args

def interact(args, data, dim, t):
    R = RipsInteract(data, dim, t)
    last = None
    while True:
        cid = R.addevent(args.fun) #, args.prime)
        if last != None:
            R.OBJ.last = last
            key = R.OBJ.query(last)
            if key: R.OBJ.plot(key)
        input = raw_input(' > ')
        if input in ['exit', 'quit', 'e', 'q']: break
        elif input == 'save': query_axis(R.fig, R.ax)
        elif ' ' in input:
            if 'save' in input:
                cmds = input.split(' ')
                i = int(cmds[1])
                if len(cmds) == 2:
                    fname = raw_input(': save ax[%d] as ' % i)
                    save_axis(R.fig, R.ax[i], fname)
                else:
                    save_axis(R.fig, R.ax[i], cmds[2])
            args = process_cmd(args, input)
        else: continue
        R.OBJ.fig.canvas.mpl_disconnect(cid)
        map(lambda a: a.cla(), R.OBJ.ax)
        last = R.OBJ.last
    return R
