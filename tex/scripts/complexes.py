from matplotlib.patches import Circle, Wedge
import matplotlib.patheffects as pfx
from scipy.spatial import Delaunay
from itertools import combinations
import matplotlib.pyplot as plt
import numpy.linalg as la
import dionysus as dio
import numpy as np
from tda import save_pkl, load_pkl
import sys

#  import SimpleLineShadow, Normal, withSimplePatchShadow


plt.ion()
ax = plt.subplot(111)

def draw_poly(s, shadow=True, **kw):
    if shadow:
        kw['path_effects'] = [pfx.withSimplePatchShadow()]
    ax.add_patch(plt.Polygon(s, **kw))

def draw_wedge(p, r, **kw):
    kw['path_effects'] = [pfx.withSimplePatchShadow()]
    ax.add_patch(Wedge(p, r, 0, 360, **kw))

def get_edges(s, c, t, op=lambda a,b: a > b):
    edges = []
    # if not all(op(la.norm(p - c), t) for p in s):
    for e in combinations(s, 2):
        if all(op(la.norm(x - c), t) for x in e):
            edges.append(e)
    return np.array(edges)

def all_edges(S, c, t, op=lambda a,b: a > b):
    x = [get_edges(s, c, t, op) for s in S]
    return np.unique(np.vstack(filter(lambda e: len(e), x)), axis=0)

# x0 = np.random.rand(1000, 2)
# c = x0.sum(axis=0) / len(x0)
# x1 = filter(lambda p: la.norm(p - c) <= 0.35, x0)
# x2 = filter(lambda p: la.norm(p - c) >= 0.15, x1)
# x = np.array(x2)
# K = Delaunay(x)
# S0 = x[K.simplices]
#
# S = filter(lambda s: max(la.norm(a - b) for a, b in combinations(s, 2)) <= 0.15, S0)

c, x, S = load_pkl('boundary.pkl')
ax.set_xlim(c[0] - 0.4, c[0] + 0.4)
ax.set_ylim(c[1] - 0.4, c[1] + 0.4)

E = np.unique(np.vstack([list(combinations(s, 2)) for s in S]), axis=0)

outer = np.array(filter(lambda p: la.norm(p - c) > 0.3, x))
inner = np.array(filter(lambda p: la.norm(p - c) < 0.19, x))

Souter = np.array(filter(lambda t: all(la.norm(p - c) > 0.3 for p in t), S))
Sinner = np.array(filter(lambda t: all(la.norm(p - c) < 0.19 for p in t), S))
Eouter = all_edges(S, c, 0.3, lambda a, b: a > b)
Einner = all_edges(S, c, 0.19, lambda a, b: a < b)

# ax.scatter(x[:,0], x[:,1], s=5, c='black', zorder=4)
# # ax.plot(x[:,0], x[:,1], 'o', c='black', zorder=4, markersize=3, path_effects=[pfx.withSimplePatchShadow()])
# map(lambda e: ax.plot(e[:,0], e[:,1], linewidth=0.75, color='black', zorder=3), E) #, path_effects=[pfx.withSimplePatchShadow()]), E)
# map(lambda s: draw_poly(s, zorder=2, alpha=0.35, color='black', linewidth=0.), S)

draw_wedge(c, 0.37, color='blue', width=0.25, zorder=0, alpha=0.1)
# ax.add_patch(Wedge(c, 0.37, 0, 360, color='blue', width=0.25, zorder=0, alpha=0.3))
ax.add_patch(Wedge(c, 0.37, 0, 360, color='red', width=0.07, zorder=1, alpha=0.2))
ax.add_patch(Wedge(c, 0.19, 0, 360, color='red', width=0.07, zorder=1, alpha=0.2))
# # draw_wedge(c, 0.37, color='red', width=0.07, zorder=1, alpha=0.2)
# # draw_wedge(c, 0.19, color='red', width=0.07, zorder=1, alpha=0.2)

# # map(lambda s: draw_poly(s, zorder=2, alpha=0.5, color='black'), S)
# ax.scatter(outer[:,0], outer[:,1], s=10, c='red', zorder=6)
# ax.scatter(inner[:,0], inner[:,1], s=10, c='red', zorder=6)
# map(lambda s: draw_poly(s, zorder=4, alpha=0.65, color='red', linewidth=0), Souter)
# map(lambda s: draw_poly(s, zorder=4, alpha=0.65, color='red', linewidth=0), Sinner)
# map(lambda e: ax.plot(e[:,0], e[:,1], alpha=0.8, color='red', zorder=5, linewidth=1.), Eouter)
# map(lambda e: ax.plot(e[:,0], e[:,1], alpha=0.8, color='red', zorder=5, linewidth=1.), Einner)

ax.axis('off')
plt.gca().set_position([0, 0, 1, 1])
if len(sys.argv) > 1:
    print('saving %s' % sys.argv[1])
    plt.savefig(sys.argv[1], transparent=True)
