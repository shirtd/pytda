from matplotlib.patheffects import SimpleLineShadow, Normal, withSimplePatchShadow
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from matplotlib import cm
import numpy as np
import os

def init(ax, x, t=0.75):
    ax.cla()
    plt.axis('off')
    xt, yt = x.max(axis=0) * t
    ax.set_xlim(-1. * xt, 1. + xt)
    ax.set_ylim(-1. * yt, 1. + yt)
    plt.tight_layout()

def save(fname):
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(fname, transparent=True)

def plot_points(ax, x, **kw):
    kw['path_effects'] = [withSimplePatchShadow()]
    ax.plot(x[:,0], x[:,1], 'o', **kw)

def plot_edges(ax, E, **kw):
    kw['path_effects'] = [SimpleLineShadow(), Normal()]
    for e in E:
        ax.plot(e[:,0], e[:,1], **kw)

def plot_circles(ax, x, r, **kw):
    kw['path_effects'] = [withSimplePatchShadow()]
    for p in x:
        ax.add_patch(plt.Circle(p, r, **kw))

def plot_tri(ax, x, **kw):
    kw['path_effects'] = [withSimplePatchShadow()]
    ax.add_patch(plt.Polygon(x, **kw))

def cover(ax, t=0.51, name='gap', dirs=['..','figures'], ext='pdf'):
    x = np.array([[0., 0.],[1., 0.], [0.5, np.sqrt(1. - 0.5 ** 2)]])
    E = np.array([[0, 1], [1, 2], [0, 2]])

    init(ax, x)
    plot_points(ax, x, zorder=2)
    plot_edges(ax, x[E], c='black', zorder=1)
    plot_circles(ax, x, t, color='red', alpha=0.5, zorder=0)
    fname = '.'.join([os.path.join(*dirs + [name]), ext])
    save(fname)

def simplicial(ax, t=0.51, name='cech', dirs=['..','figures'], ext='pdf'):
    x = np.array([[0., 0.],[1., 0.], [0.5, np.sqrt(1. - 0.5 ** 2)]])
    E = np.array([[0, 1], [1, 2], [0, 2]])

    init(ax, x)
    plot_points(ax, x, zorder=3)
    plot_edges(ax, x[E], c='black', zorder=2)
    plot_tri(ax, x, color='black', alpha=0.5, zorder=1)
    plot_circles(ax, x, t, color='red', alpha=0.5, zorder=0)
    fname = '.'.join([os.path.join(*dirs + [name]), ext])
    save(fname)

def surface(ax, dirs=['..','figures'], ext='pdf'):
    fpath = os.path.join(*dirs)
    x = np.arange(-np.pi, np.pi, 0.1)
    y = np.arange(-np.pi, np.pi, 0.1)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    Z = (Z - Z.min()) / (Z.max() - Z.min())

    ax.cla()
    ax.axis('off')
    ax.imshow(Z, cmap=cm.coolwarm, origin='lower')
    plt.tight_layout()
    save(os.path.join(fpath, 'fgrid.pdf'))

    p = 63. * np.random.rand(1000, 2)
    X, Y = np.meshgrid(range(63), range(63))
    points = np.stack((X, Y), axis=2).reshape(-1, 2)
    values = [Z[i, j] for i, j in points]
    z = griddata(points, values, p, method='cubic')

    ax.cla()
    ax.axis('off')
    ax.scatter(p[:,0],p[:,1], c=z, s=5, cmap=cm.coolwarm)
    plt.tight_layout()
    save(os.path.join(fpath, 'fsample.pdf'))

    ax.cla()
    ax.axis('off')
    ax.scatter(p[:,0],p[:,1], c='black', zorder=1, alpha=0.)
    K = Delaunay(p)
    for s in K.simplices:
        ax.add_patch(plt.Polygon(p[s], color=cm.coolwarm(sum(z[s]) / 3.)[:-1], alpha=0.5, zorder=0))
    plt.tight_layout()
    save(os.path.join(fpath, 'fcomplex.pdf'))

plt.ion()
# fig, ax = plt.subplots(1, 3, figsize=(11,4))
ax = plt.subplot(111)
# fig = plt.figure()
# ax = fig.gca(projection='3d')

if __name__ == '__main__':
    # dim = 2
    # jung = np.sqrt(2. * dim / (dim + 1.))
    # cover(ax, 0.51, 'include1')
    # simplicial(ax, 0.51, 'include2')
    # simplicial(ax, jung*0.51, 'include3')
    # cover(ax, 0.51, 'cover1')
    # cover(ax, 0.58, 'cover2')
    # cover(ax, 0.51, 'rips1')
    # simplicial(ax, 0.51, 'rips1')
    # simplicial(ax, 0.58, 'rips2')
    # surface(ax)
