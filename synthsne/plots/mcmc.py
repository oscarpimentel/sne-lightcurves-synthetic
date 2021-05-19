from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import fuzzytools.cuteplots.plots as cpplots
import fuzzytools.cuteplots.colors as cpc
from fuzzytools.cuteplots.utils import save_fig
import matplotlib.pyplot as plt

###################################################################################################################################################

def plot_mcmc_trace(mcmc_trace_bdict, b):
	mcmc_trace = mcmc_trace_bdict[b]
	az.plot_trace(mcmc_trace)
	#pm.traceplot(mcmc_trace)
	#pm.autocorrplot(mcmc_trace)

def plot_mcmc_prior(mcmc_prior, spm_p, b, c,
	save_filedir=None,
	n=100,
	):
	prior_exp = mcmc_prior.__repr__()
	data_dict = {
		'posterior spm-mle':mcmc_prior.samples,
	}
	title = f'{spm_p} {c}'
	fig, ax, legend_handles = cpplots.plot_hist_bins(data_dict, uses_density=1, title='e', cmap=cpc.get_cmap([b]), return_legend_patches=True)
	x = np.linspace(np.min(mcmc_prior.samples), np.max(mcmc_prior.samples), n)
	line = ax.plot(x, mcmc_prior.pdf(x), c=b, label=f'{prior_exp}')
	ax.legend(handles=legend_handles+[line[0]])
	ax.set_title(title)
	fig.tight_layout()
	save_fig(save_filedir, fig)