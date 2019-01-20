from tda.persist import *
from tda.hsn import *
from tda.dec import *
from tda import *

n, R, r, s = 100, 0.7, 0.2, 1.3
THRESH = 2 * s * max(R - r, r)
PRIME, DIM, TD = 53, 2, False

fig, ax = get_axes(1, 1)

if __name__ == '__main__':
    print('[ %d point %0.2f radius 1-sphere' % (n, R))
    data = circle(n, R, False, 0.)

    H = RipsCohomology(data, DIM, THRESH, PRIME)
    pt = H.get_point(1, 0)

    R = RipsDEC(H, pt)
    alpha, beta, h = R.plot_hodge(ax)
