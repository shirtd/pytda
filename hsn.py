from tda.plot import *
from tda.hsn import *
from tda import *

N, PLOT = 3025, True
DIM, THRESH = 2, np.sqrt(2)
DELTA, EXP, BOUND = 0.03, -7., 0.6
ALPHA, BETA = THRESH, 3 * THRESH

fig, ax = get_axes(1, 3, figsize=(11, 4))

if __name__ == '__main__':
    res = load_args('data')
    X = res[0] if len(res) else grf(EXP, N)
    net = HSN(X, DIM, BOUND, ALPHA, BETA, ax if PLOT else [], fig=fig if PLOT else None)

    res, HRD, H0 = net.tcc()
    print(' | D \ B^%0.2f is%s covered' % (2 * ALPHA, '' if res else ' not'))

    if not PLOT:
        net.plot(ax)
        plt.show(False)

    fname, dout = query_save(net.net_dict(), coverage=res)
