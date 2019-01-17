# from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

TD = False

def eqax(axis, lim=1.):
    axis.set_xlim(-lim, lim)
    axis.set_ylim(-lim, lim)
    if TD:
        axis.set_zlim(-lim, lim)
    return axis

plt.ion()
fig = plt.figure(1, figsize=(12, 4))
if TD:
    from mpl_toolkits.mplot3d import Axes3D
    ax = map(eqax, [plt.subplot(131+i, projection='3d') for i in range(3)])
else:
    ax = map(eqax, [plt.subplot(131+i) for i in range(3)])
plt.tight_layout()

fig2 = plt.figure(2)
ax2 = plt.subplot(111)
plt.tight_layout()

def plot_dgm(axis, D, dim=1):
    dgm = np.array([[p.birth, p.death] for p in D[dim]])
    axis.plot([0, 1], [0, 1], c='black', alpha=0.5)
    axis.scatter(dgm[:,0], dgm[:,1], s=5)

def plot_points(axis, X, *args, **kwargs):
    print(' | plotting %d vertices' % len(X))
    if TD:
        axis.scatter(X[:,0], X[:,1], X[:,2], *args, **kwargs)
    else:
        axis.scatter(X[:,0], X[:,1], *args, **kwargs)

def plot_edges(axis, E, thresh=-np.Inf, *args, **kwargs):
    if not isinstance(E, tuple):
        print(' | plotting %d edges' % len(E))
        for e in E:
            if TD:
                axis.plot(e[:,0], e[:,1], e[:,2], *args, **kwargs)
            else:
                axis.plot(e[:,0], e[:,1], *args, **kwargs)
    else:
        E, z = E
        print(' | plotting %d edges' % len(E))
        # thresh = kwargs['thresh'] if 'thresh' in kwargs else -np.Inf
        color = cm.ScalarMappable(Normalize(z.min(), z.max()), cm.rainbow)
        for e, a in zip(E, z):
            if a >= thresh:
                if TD:
                    axis.plot(e[:,0], e[:,1], e[:,2], c=color.to_rgba(a), *args, **kwargs)
                else:
                    axis.plot(e[:,0], e[:,1], c=color.to_rgba(a), *args, **kwargs)
        color.set_array(filter(lambda a: a >= thresh, z))
        plt.colorbar(color, ax=axis)
