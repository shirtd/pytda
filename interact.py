#!/usr/bin/env python
from tda.interact import *

if __name__ == '__main__':
    args = parser.parse_args()
    data, dim, t = double_circle(200), 2, np.sqrt(2) * 2
    R = interact(args, data, dim, t)
