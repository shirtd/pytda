import dionysus as dio
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

plt.ion()
ax = plt.subplot(111)
plt.tight_layout()

x = np.array([[0., 0.],
            [1., 0.],
            [1., 1.],
            [0., 1.]])
y = np.array([[0.5, 0.25],
            [0.75, 0.5],
            [0.5, 0.75],
            [0.25, 0.5]])

X = np.vstack((x, y))
E = np.array([[0, 1], [1, 2],
            [2, 3], [3, 0],
            [0, 4], [1, 4],
            [1, 5], [2, 5],
            [2, 6], [3, 6],
            [3, 7], [0, 7],
            [4, 5], [5, 6],
            [6, 7], [7, 4]])

T = np.array([[0, 1, 4],
            [1, 2, 5],
            [2, 3, 6],
            [3, 0, 7],
            [0, 7, 4],
            [1, 4, 5],
            [2, 5, 6],
            [3, 6, 7]])

F = dio.Filtration()
for i, v in enumerate(X):
    F.append(dio.Simplex([i], len(F)))
for i, e in enumerate(E):
    F.append(dio.Simplex(e, len(F)))
for i,t in enumerate(T):
    F.append(dio.Simplex(t, len(F)))
F.append(dio.Simplex([4, 6], len(F)))
F.append(dio.Simplex([4, 5, 6], len(F)))
F.append(dio.Simplex([6, 7, 4], len(F)))
F.sort()

CH = dio.cohomology_persistence(F, 2, True)
CD = dio.init_diagrams(CH, F)

pt = max(CD[1], key=lambda p: p.death)
cocycle = CH.cocycle(pt.data)
l = np.array([list(F[c.index]) for c in cocycle])

ax.plot(X[:,0], X[:,1], 'o', zorder=3)
map(lambda e: ax.plot(e[:,0], e[:,1], c='black', zorder=1), X[E])
map(lambda t: ax.add_patch(plt.Polygon(t, color='black', alpha=0.5, zorder=0)), X[T])
map(lambda e: ax.plot(e[:,0], e[:,1], c='red', zorder=2), X[l])

H = dio.homology_persistence(F, 2)
D = dio.init_diagrams(H, F)

# pt = max(D[1], key=lambda p: p.death - p.birth)
for pt in D[1]:
    cycle = H[H.pair(pt.data)]
    l = np.array([list(F[c.index]) for c in cycle])
    ax.cla()
    plt.tight_layout()
    ax.plot(X[:,0], X[:,1], 'o', zorder=3)
    map(lambda e: ax.plot(e[:,0], e[:,1], c='black', zorder=1), X[E])
    map(lambda t: ax.add_patch(plt.Polygon(t, color='black', alpha=0.5, zorder=0)), X[T])
    map(lambda e: ax.plot(e[:,0], e[:,1], c='red', zorder=2), X[l])
    plt.show(False)
    raw_input()
