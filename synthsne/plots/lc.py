from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import matplotlib.pyplot as plt
from lchandler.plots.lc import plot_lightcurve
import random
from fuzzytools.strings import latex_bf_alphabet_count

COLOR_DICT = _C.COLOR_DICT
FIGSIZE = (10,5)
DPI = 200

###################################################################################################################################################

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
	survey = lcset_info['survey']

	fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
	synth_curves_plot_max = len(new_smooth_lcojbs) if synth_curves_plot_max is None else synth_curves_plot_max

	for k in range(0, synth_curves_plot_max):
		new_smooth_lcojb = new_smooth_lcojbs[k]
		for b in band_names:
			ax.plot(new_smooth_lcojb.get_b(b).days, new_smooth_lcojb.get_b(b).obs, lw=lw, c=COLOR_DICT[b], alpha=alpha)
			ax.plot([None], [None], lw=1, c=COLOR_DICT[b], label=f'{b} SPM posterior sample' if k==0 else None)

	idx = random.randint(0, len(new_smooth_lcojbs)-1)
	for b in band_names:
		plot_lightcurve(ax, lcobj, b, label=f'{b} obs')
		plot_lightcurve(ax, new_lcobjs[idx], b, label=f'{b} obs')

	ax.grid(alpha=0.0)
	title = ''
	title += f'SN multi-band light-curve generation; method={method}; $k_s$={synth_curves_plot_max}'+'\n'
	# title += f'set={survey} [{lcset_name.replace(".@", "")}]'+'\n'
	title += f'{latex_bf_alphabet_count(class_names.index(class_names[lcobj.y]))} obj={lcobj_name} [{class_names[lcobj.y]}]'+'; '+'; '.join([f'{b}-error={trace_bdict[b].get_xerror()}' for b in band_names])+'\n'
	ax.set_title(title[:-1])
	ax.legend(loc='upper right')
	ax.set_xlabel('time [days]')
	ax.set_ylabel('observation [flux]')

	fig.tight_layout()
	return fig, ax