#!/usr/bin/env python
from tda.interact import *

if __name__ == '__main__':
    data = double_circle(400, (1., 0.7), False, 0.1)
    dim, t = 2, np.sqrt(2) * 2
    R = RipsCommand(data, dim, t)
    R.do_homology(None)
    R.do_circular(None)
    R.cmdloop()
