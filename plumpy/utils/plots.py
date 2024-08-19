'''

'''

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

def save_plot(plot_path, name, format='png'):
    if plot_path is not None:
        if not Path(plot_path).exists(): Path(plot_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path / Path(name), format=format, dpi=160, transparent=True, bbox_inches='tight')
        plt.close()


def plot_data(data, events, plot_chan=5, seg_id=0, units=None):
    ch_idx = data["elec_ids"].index(plot_chan)
    t = data["data_headers"][seg_id]["Timestamp"] / data["samp_per_s"]
    t2 = np.array(events['digital_events']['TimeStamps']) / data["samp_per_s"]
    plt.plot(t, data["data"][seg_id][ch_idx])
    plt.axis([t[0], t[-1], min(data["data"][seg_id][ch_idx]), max(data["data"][seg_id][ch_idx])])
    plt.locator_params(axis="y", nbins=20)
    plt.xlabel("Time (s)")
    if units: plt.ylabel("Output (" + units[ch_idx] + ")")
    plt.title(plot_chan)
    plt.vlines(x=t2, ymin=-400, ymax=400, color='black')


def plot_psd(signal, fmax):
    import seaborn as sns
    with sns.plotting_context('poster'):
        signal.plot_psd(fmin=1, fmax=fmax)
        plt.tight_layout()
##
def plot_signals_on_grid(signals, grid, outliers=[-1], ymin=-2000, ymax=2000):
    import seaborn as sns
    with sns.plotting_context('poster', font_scale=1):
        fig = plt.figure(figsize=(14, 16), layout="constrained")
        spec = fig.add_gridspec(nrows=16, ncols=8)
        for i in range(16):
            for j in range(8):
                ax = fig.add_subplot(spec[i, j])
                if grid[i, j]-1 in outliers:
                    color = 'red'
                else:
                    color = '#1f77b4'
                plt.plot(signals[grid[i,j]-1], color=color)
                plt.title(grid[i, j])
                plt.ylim(ymin, ymax)
                ax.set(xticklabels=[])  # remove the tick labels
                ax.tick_params(bottom=False)
                if not (i == 0 and j == 0):
                    ax.set(yticklabels=[])  # remove the tick labels
                    ax.tick_params(left=False)
##
def plot_on_grid(grid, values, colormap='viridis', ax=None, label='', xmin=None, xmax=None, show_cbar=True):
    if ax is None:
        plt.figure(figsize=(4, 6), dpi=160)
        ax = plt.gca()
    if not xmin:
        xmin = np.min(values)
    if not xmax:
        xmax = np.max(values)
    grid_values = values[grid - 1]
    im = ax.imshow(grid_values, aspect='auto', cmap=colormap, vmin=xmin, vmax=xmax,
                   extent=[0, grid.shape[-1], 0, grid.shape[0]])
    for (i, j), k in np.ndenumerate(np.flipud(grid - 1)):
        ax.text(j + .6, i + .6, k + 1, ha='center', va='center', color=(.5, .5, .5), fontsize=8)
    if show_cbar:
        cbar = plt.gcf().colorbar(im)
        cbar.ax.set_ylabel(label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

##
