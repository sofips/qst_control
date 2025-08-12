import os
import numpy as np

dimensions = [8,16,32,64]

for dim in dimensions:
    dirname = 'n{}_gpu'.format(dim)
    cmd = 'nice -20 python3 exp_gen.py {} {} >out{}.out &'.format(dim, dirname, dim)
    os.system(cmd)
    os.system('bg')
    os.system('disown')