import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import pymrt as mrt
import pymrt.plot



arr = np.random.randint(0, 100, (2, 3, 4))

mrt.plot.sample2d(
    arr, cbar_kws=dict(ticks=(0, 100)), cbar_txt='arb. units')

plt.tight_layout()
plt.show()
