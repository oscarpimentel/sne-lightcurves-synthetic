from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
from flamingchoripan.cuteplots.utils import save_fig
from lchandler.plots.lc import plot_lightcurve

###################################################################################################################################################

def get_errors_txt(trace, b):
	mean = trace.get_errors_mean()
	txt = f'{b}-error: '
	if mean is None:
		return txt+'-'
	else:
		return txt+f'{mean:.2f}$\pm${trace.get_errors_std():.1f}'

def get_error_txt(trace, b, k):
	error = trace.get_error(k)
	txt = f'{b}-error: '
	if error is None:
		return txt+'-'
	else:
		return txt+f'{error:.2f}'

def plot_synthetic_samples(lcdataset, set_name:str, method, lcobj_name, new_lcobjs, new_smooth_lcojbs,
	trace_bdict=None,
	figsize:tuple=(13,6),
	lw=1.5,
	save_filedir=None,
	):
	lcset = lcdataset[set_name]
	fig, axs = plt.subplots(1, 2, figsize=figsize)
	band_names = lcset.band_names
	lcobj = lcset[lcobj_name]
	idx = 0

	###
	ax = axs[0]
	for b in band_names:
	    plot_lightcurve(ax, lcobj, b, label=f'{b} observation')
	    for k,new_smooth_lcojb in enumerate(new_smooth_lcojbs):
	        label = f'{b} posterior pm-sample' if k==0 else None
	        ax.plot(new_smooth_lcojb.get_b(b).days, new_smooth_lcojb.get_b(b).obs, alpha=0.15, lw=1, c=C_.COLOR_DICT[b]); ax.plot(np.nan, np.nan, lw=1, c=C_.COLOR_DICT[b], label=label)
	ax.grid(alpha=0.5)
	title = f'multiband light curve & parametric model samples\n'
	title += f'method: {method} - '+' - '.join([get_errors_txt(trace_bdict[b], b) for b in band_names])+'\n'
	title += f'survey: {lcset.survey}/{set_name} - obj: {lcobj_name}- class: {lcset.class_names[lcobj.y]}'
	ax.set_title(title)
	ax.legend(loc='upper right')
	ax.set_ylabel('obs[flux]')
	ax.set_xlabel('days')

	###
	ax = axs[1]
	for b in band_names:
	    plot_lightcurve(ax, lcobj, b, label=f'{b} observation')
	    for k,new_lcobj in enumerate([new_lcobjs[idx]]):
	        plot_lightcurve(ax, new_lcobj, b, label=f'{b} observation' if k==0 else None)
	        
	ax.grid(alpha=0.5)
	title = f'multiband light curve & synthetic curve example\n'
	title += f'method: {method} - '+' - '.join([get_error_txt(trace_bdict[b], b, idx) for b in band_names])+'\n'
	title += f'survey: {lcset.survey}/{set_name} - obj: {lcobj_name}- class: {lcset.class_names[lcobj.y]}'
	ax.set_title(title)
	ax.legend(loc='upper right')
	#ax.set_ylabel('obs [flux]')
	ax.set_xlabel('days')

	fig.tight_layout()
	save_fig(fig, save_filedir)