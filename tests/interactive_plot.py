import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import pymrt as mrt
import pymrt.plot


def _ax_sizes(ax):
    bbox = ax.get_window_extent().transformed(
        ax.figure.dpi_scale_trans.inverted())
    sizes = tuple(size * ax.figure.dpi for size in (bbox.width, bbox.height))
    return sizes


values = 0, 1000
arr = np.random.randint(values[0], values[1], (2, 3, 4))

# mrt.plot.sample2d(
#     arr, cbar_kws=dict(ticks=(0, 100)), cbar_txt='arb. units')

from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# fig, ax = plt.subplots(1, 1)

pax = ax.imshow(arr, vmin=values[0], vmax=values[1])

# cbar_kws=dict(ticks=(-1, -2))
ax_sizes_pt = _ax_sizes(ax)
print(fig, ax.figure)
cbar_kws = dict(ticks=values)
cbar_txt = 'arb. units'
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = ax.figure.colorbar(pax, cax=cax, **dict(cbar_kws))
# cbar = ax.figure.colorbar(plot, ax=ax, **dict(cbar_kws))
if cbar_txt is not None:
    only_extremes = 'ticks' in cbar_kws and len(cbar_kws['ticks']) == 2
    if only_extremes:
        print(_ax_sizes(cax))
        # cbar.ax.text(
        #     2.0, 0.5, cbar_txt, fontsize='medium', rotation=90,
        #     va='center', ha='left', transform=cbar.ax.transAxes)
        cbar.set_label(cbar_txt, labelpad=-ax_sizes_pt[0] * 0.075)
        cbar.ax.yaxis.set_label_position('left')
    else:
        cbar.set_label(cbar_txt)

plt.tight_layout()
plt.show()
