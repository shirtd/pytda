from matplotlib.patches import Circle, Wedge
import matplotlib.patheffects as pfx
from scipy.spatial import Delaunay
from itertools import combinations
import matplotlib.pyplot as plt
import numpy.linalg as la
import dionysus as dio
import numpy as np
import sys

plt.ion()
ax = plt.subplot(111)
ax.set_xlim(-1.6, 3.2)
ax.set_ylim(-1.75, 3.2)

def draw_poly(s, shadow=True, **kw):
    if shadow:
        kw['path_effects'] = [pfx.withSimplePatchShadow()]
    ax.add_patch(plt.Polygon(s, **kw))

def draw_polys(T, shadow=True, **kw):
    map(lambda t: draw_poly(t, shadow, **kw), T)

def draw_circle(p, r, shadow=True, **kw):
    if shadow:
        kw['path_effects'] = [pfx.withSimplePatchShadow()]
    ax.add_patch(Circle(p, r, **kw))

def draw_circles(x, r, shadow=True, **kw):
    map(lambda p: draw_circle(p, r, shadow, **kw), x)

def draw_wedge(p, r, shadow=True, **kw):
    if shadow:
        kw['path_effects'] = [pfx.withSimplePatchShadow()]
    ax.add_patch(Wedge(p, r, 0, 360, **kw))

def draw_points(x, shadow=True, **kw):
    if shadow:
        kw['path_effects'] = [pfx.withSimplePatchShadow()]
    ax.plot(x[:,0], x[:,1], 'o', markersize=5, **kw)

def draw_edge(e, shadow=True, **kw):
    if shadow:
        kw['path_effects'] = [pfx.SimpleLineShadow(), pfx.Normal()]
    ax.plot(e[:,0], e[:,1], **kw)

def draw_edges(E, shadow=True, **kw):
    map(lambda e: draw_edge(e, shadow, **kw), E)

x = np.array([[0.21, -0.15],
            [-0.1, 0.9],
            [0.4, 0.4],
            [-0.125, 0.3],
            [0.15, 0.8],
            [0.35, 1.35],
            [0.9, 1.7],
            [1.5, 1.5],
            [1.7, 0.9],
            [1.45, 0.3],
            [0.83, 0.1]])

t = 1.4

fE = lambda i, j:  i != j and la.norm(x[i] - x[j]) <= t
E = np.array([[i, j] for i in range(len(x)) for j in range(i) if fE(i, j)])
fT = lambda i, j, k:  all(e in E.tolist() for e in ([i, j], [j, k], [i, k]))
T = np.array([[i, j, k] for i in range(len(x)) for j in range(len(x)) for k in range(len(x)) if fT(i, j, k)])


draw_points(x, c='black', zorder=3)
draw_edges(x[E], c='black', zorder=2)
draw_circles(x, t, color='red', alpha=0.05, zorder=0)
draw_polys(x[T], color='black', alpha=0.15, zorder=1)

ax.axis('off')
plt.gca().set_position([0, 0, 1, 1])
if len(sys.argv) > 1:
    print('saving %s' % sys.argv[1])
    plt.savefig(sys.argv[1], transparent=True)
