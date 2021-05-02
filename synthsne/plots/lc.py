from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
from lchandler.plots.lc import plot_lightcurve
import random

###################################################################################################################################################

def plot_synthetic_samples(lcobj_name, lcobj, lcset_name, lcset_info, method, new_lcobjs, new_smooth_lcojbs,
	synth_curves_plot_max=None,
	trace_bdict=None,
	figsize:tuple=(8,12),
	lw=1.5,
	):
	band_names = lcset_info['band_names']
	class_names = lcset_info['class_names']
	survey = lcset_info['survey']

	fig, axs = plt.subplots(2, 1, figsize=figsize)
	synth_curves_plot_max = len(new_smooth_lcojbs) if synth_curves_plot_max is None else synth_curves_plot_max

	ax = axs[0]
	for b in band_names:
		plot_lightcurve(ax, lcobj, b, label=f'{b} obs')
		for k in range(0, synth_curves_plot_max):
			new_smooth_lcojb = new_smooth_lcojbs[k]
			label = f'{b} SPM posterior samples' if k==0 else None
			ax.plot(new_smooth_lcojb.get_b(b).days, new_smooth_lcojb.get_b(b).obs, alpha=0.25, lw=1, c=C_.COLOR_DICT[b]); ax.plot(np.nan, np.nan, lw=1, c=C_.COLOR_DICT[b], label=label)
	ax.grid(alpha=0.5)
	title = ''
	title += f'SNe multiband light curve & parametric model samples\n'
	title += f'survey={survey}-{"".join(band_names)} [{lcset_name}] - obj={lcobj_name} [{class_names[lcobj.y]}]'+'\n'
	title += ' - '.join([f'method={method}']+[f'{b}-error={trace_bdict[b].get_xerror()}' for b in band_names])+'\n'
	ax.set_title(title[:-1])
	ax.legend(loc='upper right')
	ax.set_ylabel('observations [flux]')
	#ax.set_xlabel('time[days]')

	###
	ax = axs[1]
	idx = random.randint(0, len(new_smooth_lcojbs)-1)
	for b in band_names:
		plot_lightcurve(ax, lcobj, b, label=f'{b} obs', alpha=0.25)
		for k,new_lcobj in enumerate([new_lcobjs[idx]]):
			plot_lightcurve(ax, new_lcobj, b, label=f'{b} obs' if k==0 else None)
			
	ax.grid(alpha=0.5)
	title = ''
	title += ' - '.join([f'method={method}']+[f'{b}-error={trace_bdict[b].get_xerror_k(idx).set_repr_pm(False)}' for b in band_names])+'\n'
	ax.set_title(title[:-1])
	ax.legend(loc='upper right')
	ax.set_ylabel('observations [flux]')
	ax.set_xlabel('time [days]')

	fig.tight_layout()
	return fig, axs