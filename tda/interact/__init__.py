from ..plot import *
from cmd import Cmd

class InteractCmd(Cmd, Interact):
    def __init__(self, fig, ax, classes, **kw):
        Interact.__init__(self, fig, ax, classes, **kw)
        self.last, self.cid = None, None
        Cmd.__init__(self)
        self.prompt = '> '
    # def parse_args(self, args):
    #     cmds = args.split(',')
    #     kvs = map(lambda s: s.split(), cmds)
    #
    def function(self, fun, **kw):
        self.fun = fun
        if self.cid != None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.OBJ.clear()
            self.last = self.OBJ.last
        self.cid = self.addevent(fun, **kw)
        if self.last != None:
            self.OBJ.last = self.last
            key = self.OBJ.query(self.last)
            if key: self.OBJ.plot(key)
        plt.show(False)
    def do_prime(self, p):
        prime, cache = self.prime, self.cache
        try:
            self.prime, self.cache = int(p), {}
            self.function(self.fun)
        except:
            print(' ! invalid prime %s' % p)
            self.prime, self.cache = prime, cache
    def do_save(self, args):
        try:
            i, fname = args.split()
            save_axis(self.fig, self.ax[int(i)], fname)
        except:
            print(' ! invalid args %s' % args)
    def do_fun(self, fun):
        try:
            self.function(fun)
        except:
            print(' ! invalid function %s' % fun)
    def do_exit(self, args): raise SystemExit
    def do_EOF(self, line): return True
