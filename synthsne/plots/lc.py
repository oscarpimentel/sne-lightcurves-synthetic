import numpy as np
import matplotlib.pyplot as plt
import fuzzytools as ft
from lchandler.plots.lc import plot_lightcurve

from . import _C


COLOR_DICT = _C.COLOR_DICT
FIGSIZE = (8, 4)
DPI = 200


def plot_synthetic_samples(lcobj_name, lcobj, lcset_name, lcset_info, method, new_lcobjs, new_smooth_lcojbs,
                           synth_curves_plot_max=None,
                           trace_bdict=None,
                           figsize=FIGSIZE,
                           dpi=DPI,
                           lw=.1,
                           alpha=.75,
                           ):
    band_names = lcset_info['band_names']
    class_names = lcset_info['class_names']

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    synth_curves_plot_max = len(new_smooth_lcojbs) if synth_curves_plot_max is None else synth_curves_plot_max

    for k in range(0, synth_curves_plot_max):
        new_smooth_lcojb = new_smooth_lcojbs[k]
        for b in band_names:
            ax.plot(new_smooth_lcojb.get_b(b).days, new_smooth_lcojb.get_b(b).obs, lw=lw, c=COLOR_DICT[b], alpha=alpha)
            ax.plot([None], [None], lw=1, c=COLOR_DICT[b], zorder=-1, label=f'{b} SPM posterior sample' if k == 0 else None)

    idx = np.random.randint(0, len(new_smooth_lcojbs) - 1)
    for b in band_names:
        plot_lightcurve(ax, lcobj, b, label=f'{b} obs')
        plot_lightcurve(ax, new_lcobjs[idx], b, label=f'{b} obs')

    ax.grid(alpha=0.0)
    title = ''
    title += f'SN multi-band light-curve generation; method={method}; $k_s$={synth_curves_plot_max}\n'
    class_name = class_names[lcobj.y].replace('*', '')
    title += f'{ft.strings.latex_bf_alphabet_count(class_names.index(class_names[lcobj.y]))} obj={lcobj_name} [{class_name}]\n'
    # title += '; '.join([f'{b}-error={trace_bdict[b].get_xerror()}' for b in band_names]) + '\n'
    ax.set_title(title[:-1])
    ax.legend(loc='upper right')
    ax.set_ylabel('observation [flux]')
    ax.set_xlabel('observation-time [days]')
    fig.tight_layout()
    return fig, ax
