from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

plt.ion()

def eqax(axis, lim=1., is_3d=False):
    axis.set_xlim(-lim, lim)
    axis.set_ylim(-lim, lim)
    if is_3d:
        axis.set_zlim(-lim, lim)
    return axis

def get_axes(row=1, col=1, is_3d=False, **kw):
    if is_3d: kw['projection'] = '3d'
    fig, ax = plt.subplots(row, col, **kw)
    plt.tight_layout()
    return fig, ax

def plot_edges(axis, E, z=[], thresh=-np.Inf, **kw):
    print('[ plotting %d edges' % len(E))
    if len(z):
        color = cm.ScalarMappable(Normalize(z.min(), z.max()), cm.rainbow)
    c = map(color.to_rgba, z) if len(z) else ['black'] * len(E)
    for e, a in zip(E, c):
        kw['c'], dims = a, [e[:,i] for i in range(e.shape[1])]
        axis.plot(*dims, **kw)
    if len(z):
        color.set_array(filter(lambda a: a >= thresh, z))
        plt.colorbar(color, ax=axis)
