from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import fuzzytools.matplotlib.plots as cplots
from fuzzytools.matplotlib.utils import save_fig
from fuzzytools.strings import bf_alphabet_count

FIGSIZE_2X1 = (14, 5.5)
DPI = 200

###################################################################################################################################################

def plot_obse_samplers(lcset_name, lcset_info, obse_sampler_bdict,
	original_space:bool=1,
	pdf_scale:float=0.01,
	figsize=FIGSIZE_2X1,
	dpi=DPI,
	add_samples=0,
	save_filedir=None,
	):
	survey = lcset_info['survey']
	band_names = lcset_info['band_names']

	fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
	for kb,b in enumerate(band_names):
		ax = axs[kb]
		obse_sampler = obse_sampler_bdict[b]
		if original_space:
			label = '$p(x_{ij},\sigma_{xij})$'+f' {lcset_name} samples'
			ax.plot(obse_sampler.raw_obse, obse_sampler.raw_obs, 'k.', markersize=2, alpha=0.2); ax.plot(np.nan, np.nan, 'k.', label=label)

			### add samples
			if add_samples:
				n = int(2e3)
				to_sample = obse_sampler.raw_obs
				std = 1e-4
				new_obs = [np.random.normal(to_sample[np.random.randint(0, len(to_sample))], std) for _ in range(n)] # kde
				#x = 0.05; new_obs = np.linspace(x, x+0.001, 1000) # sanity check
				new_obse, new_obs = obse_sampler.conditional_sample(new_obs)
				ax.plot(new_obse, new_obs, 'r.', markersize=2, alpha=1); ax.plot(np.nan, np.nan, 'r.', label='$\hat{p}(x_{ij},\sigma_{xij})$ samples (KDE)')

			### rot axis
			x = np.linspace(obse_sampler.raw_obse.min(), obse_sampler.raw_obse.max(), 100)
			ax.plot(x, x*obse_sampler.m+obse_sampler.n, 'b', alpha=0.75, label='rotation axis', lw=1)
			#ax.plot(obse_sampler.lr_x, obse_sampler.lr_y, 'b.', alpha=1, markersize=4); ax.plot(np.nan, np.nan, 'b.', label='rotation axis support samples')

			ax.set_xlabel('observation error')
			ax.set_ylabel('observation' if kb==0 else None)
			ax.set_xlim([0.0, 0.05])
			ax.set_ylim([0.0, 0.3])

		else:
			label='$p(x_{ij}'+"'"+',\sigma_{xij}'+"'"+')$'+f' empirical samples'
			ax.plot(obse_sampler.obse, obse_sampler.obs, 'k.', markersize=2, alpha=0.2); ax.plot(np.nan, np.nan, 'k.', label=label)
			min_obse = obse_sampler.obse.min()
			max_obse = obse_sampler.obse.max()
			pdfx = np.linspace(min_obse, max_obse, 200)
			colors = cm.inferno(np.linspace(0, .5, len(obse_sampler.distrs)))
			min_pdfy = np.infty
			for p_idx in range(len(obse_sampler.distrs)):
				d = obse_sampler.distrs[p_idx]
				if p_idx%10==0:
					rank_ranges = obse_sampler.rank_ranges[p_idx]
					pdf_offset = rank_ranges[1] # upper of rank range
					pdfy = d['distr'].pdf(pdfx, *d['params'])
					pdfy = pdfy/pdfy.max()*pdf_scale+pdf_offset
					c = colors[p_idx]
					label = '$q(\sigma_{xij}'+"'"+'|x_{ij}'+"'"+')$ normal conditional fit'
					ax.plot(pdfx, pdfy, c=c, alpha=1, lw=1, label=label if p_idx==0 else None)
					min_pdfy = pdfy.min() if pdfy.min()<min_pdfy else min_pdfy
			
			ax.set_xlabel('rotated-flipped observation error')
			ax.set_ylabel('rotated observation' if kb==0 else None)
			#ax.set_xlim([0.0, 0.02])
			ax.set_ylim([min_pdfy, 1])

		title = ''
		title += f'{bf_alphabet_count(kb)} Joint distribution from {lcset_name}; band={b}'+'\n'
		ax.set_title(title[:-1])
		ax.legend(loc='upper left')

		### multiband colors
		[ax.spines[border].set_color(_C.COLOR_DICT[b]) for border in ['bottom', 'top', 'right', 'left']]
		[ax.spines[border].set_linewidth(2) for border in ['bottom', 'top', 'right', 'left']]

	fig.tight_layout()
	save_fig(save_filedir, fig)