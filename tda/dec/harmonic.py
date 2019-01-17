from numpy import asarray, eye, outer, inner, dot, vstack
from numpy.random import seed, rand
from numpy.linalg import norm
from . import *

def hodge_decomposition(omega):
    """
    For a given p-cochain \omega there is a unique decomposition

    \omega = d(\alpha) + \delta(\beta) (+) h

    for p-1 cochain \alpha, p+1 cochain \beta, and harmonic p-cochain h.

    This function returns (non-unique) representatives \beta, \gamma, and h
    which satisfy the equation above.

    Example:
        #decompose a random 1-cochain
        sc = SimplicialComplex(...)
        omega = sc.get_cochain(1)
        omega.[:] = rand(*omega.shape)
        (alpha,beta,h) = hodge_decomposition(omega)

    """
    print('[ constructing hodge decomposition')
    sc, p = omega.complex, omega.k
    alpha = sc.get_cochain(p - 1)
    beta  = sc.get_cochain(p + 1)
    print(' | solving for alpha')
    A = delta(d(sc.get_cochain_basis(p - 1))).v
    b = delta(omega).v
    alpha.v = cg( A, b, tol=1e-8 )[0]
    print(' | solving for beta')
    A = d(delta(sc.get_cochain_basis(p + 1))).v
    b = d(omega).v
    beta.v = cg( A, b, tol=1e-8 )[0]
    # Solve for h
    h = omega - d(alpha) - delta(beta)
    return alpha, beta, h

def hodge2(rc, c):
    rc, p, x = c.complex, c.k, c.v
    cmplx = rc.chain_complex()  # boundary operators [ b0, b1, b2 ]
    b1 = cmplx[1].astype(float)  # edge boundary operator
    b2 = cmplx[2].astype(float)  # face boundary operator
    x = c.v

    # Decompose x using discrete Hodge decomposition
    alpha = rc.get_cochain(p - 1)
    beta = rc.get_cochain(p + 1)
    h = rc.get_cochain(p)

    a = cg( b1 * b1.T, b1 * x, tol=1e-8)[0]
    b = cg( b2.T * b2, b2.T * x, tol=1e-8)[0]
    h.v = x - (b1.T * a) - (b2 * b) # harmonic component of x
    alpha.v, beta.v = a, b
    # h /= abs(h).max() # normalize h
    return alpha, beta, h

def ortho(A):
    """Separates the harmonic forms stored in the rows of A using a heuristic
    """
    A = asarray(A)

    for i in range(A.shape[0]):
        j = abs(A[i]).argmax()
        v = A[:,j].copy()
        if A[i,j] > 0:
            v[i] += norm(v)
        else:
            v[i] -= norm(v)
        Q = eye(A.shape[0]) - 2 * outer(v, v) / inner(v, v)
        A = dot(Q, A)
    return A

def get_harmonic(X, F, H, D, n=4):
    pts = sorted(D[1], key=lambda p: p.death - p.birth, reverse=True)
    n = len(pts) if len(pts) < n else n
    HF, SC = [], []  # harmonic forms
    for i in range(n):
        pt = pts[i]
        R = to_dec(X, F, pt, 2)
        c = get_cochain(F, R, H, pts[i])
        beta, gamma, h = hodge_decomposition(c)
        # beta, gamma, h = hodge2(R, c)
        h = h.v
        for v in HF:
            h -= inner(v,h) * v
        h /= norm(h)
        HF.append(h)
        SC.append(R)
    return ortho(vstack(HF)), SC, pts

def illinois(c):
    rc, p, x = c.complex, c.k, c.v
    d = dec.d(rc.get_cochain_basis(p - 1)).v
    A = d.T * rc[p].star * d
    b = -1 * d.T * rc[p].star * x
    alpha = rc.get_cochain(p - 1)
    alpha.v = cg(A, b, tol=1e-8)[0]
    return alpha
