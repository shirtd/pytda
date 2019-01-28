from tda.interact.hsn import *
# from tda.plot import *
# from tda.hsn import *
from tda import *

N, PLOT = 1024, True
NOISE = 0. # 1.5e1 / np.sqrt(N)
DIM, THRESH = 2, (np.sqrt(2) + NOISE)
DELTA, EXP, BOUND = 0.03, -11., 0.3
ALPHA, BETA = THRESH, 3 * THRESH

if __name__ == '__main__':
    res = load_args('data')
    X = res[0] if len(res) else grf(EXP, N)

    R = HSNInteract(X, DIM, BOUND, ALPHA, BETA, NOISE)
    R.cmdloop()
