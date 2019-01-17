from interact import *
from util import *
import argparse

parser = argparse.ArgumentParser(description='interactive cycles, cocycles, and circular coordinates.')
parser.add_argument('dset', default='mnist', nargs='?', help='data set. default: mnist.')
parser.add_argument('-c','--c', default=0, type=int, help='image class. default: 0.')
parser.add_argument('-i','--i', default=0, type=int, help='image index. default: 0.')
parser.add_argument('-f','--fun', default='homology', help='persistence function. default: homology.')
parser.add_argument('-p','--prime', default=2, type=int, help='field coefficient. default: 2 (11 if fun = circular).')

def interact(args):
    shape = SHAPE[args.dset]
    X, y = load_data(args.dset)
    G = group_data(X, y)
    i, last = 0, None
    i, last = 0, None
    while i < len(G[args.c]):
        cid, obj = addevent(anevent, G[args.c][i], args.fun, args.prime, args.dset)
        if last != None:
            obj.last = last
            key = obj.query(last)
            if key:
                obj.plot_image(key)
                plt.show()
        input = raw_input(' > ')
        if input in ['exit', 'quit', 'e', 'q']:
            break
        elif ' ' in input:
            cmds = input.split(',') if ',' in input else [input]
            for cmd in cmds:
                k, v = [c for c in cmd.split(' ') if len(c) > 0]
                if 'class' in k:
                    args.c, i = int(v), 0
                elif 'data' in k:
                    args.dset = v
                    shape = SHAPE[args.dset]
                    X, y = load_data(args.dset)
                    G, i = group_data(X, y), 0
                elif 'fun' in k or 'function' in k:
                    args.fun = v
                elif 'prime' in k:
                    args.prime = int(v)
                else:
                    print('\tunknown command %s' % cmd)
        else:
            i += 1
        fig.canvas.mpl_disconnect(cid)
        map(lambda a: a.cla(), ax)
        last = obj.last
