from interact import *
from .. import *
import argparse

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
        elif ' ' in input:
            args = process_cmd(args, input)
        else: continue
        R.OBJ.fig.canvas.mpl_disconnect(cid)
        map(lambda a: a.cla(), R.OBJ.ax)
        last = R.OBJ.last
    return R
