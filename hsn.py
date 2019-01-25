from tda.plot import *
from tda.hsn import *
from tda import *

N, PLOT = 4900, True
NOISE = 1.5e1 / np.sqrt(N)
DIM, THRESH = 2, (np.sqrt(2) + NOISE) # * (1 + 1e-5)
DELTA, EXP, BOUND = 0.03, -5., 0.2
ALPHA, BETA = THRESH, 3 * THRESH

fig, ax = get_axes(1, 3, figsize=(11, 4))

if __name__ == '__main__':
    res = load_args('data')
    X = res[0] if len(res) else grf(EXP, N)
    net = HSN(X, DIM, BOUND, ALPHA, BETA, NOISE, *(fig, ax) if PLOT else (None, []))

    res, HRD, H0 = net.tcc()
    print(' | D \ B^%0.2f is%s covered' % (2 * ALPHA, '' if res else ' not'))

    if not PLOT:
        net.plot(ax)
        plt.show(False)

    fname, dout = query_save(net.net_dict(), coverage=res)
