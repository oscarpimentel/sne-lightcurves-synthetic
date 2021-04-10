from __future__ import print_function
from __future__ import division
from . import C_
import cProfile

import numpy as np
from flamingchoripan.progress_bars import ProgressBar
from flamingchoripan.files import save_pickle, filedir_exists
from .synthetic_curves import get_syn_sne_generator
from ..plots.lc import plot_synthetic_samples
from flamingchoripan.lists import get_list_chunks
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

###################################################################################################################################################

def is_in_column(lcobj_name, sne_specials_df, column):
	if sne_specials_df is None:
		return False
	return lcobj_name in list(sne_specials_df[column].values)

def generate_synthetic_samples(lcobj_name, lcobj, lcset_name, lcset_info, obse_sampler_bdict, uses_estw, save_rootdir,
	method=None,
	synthetic_samples_per_curve:float=4,
	sne_specials_df=None,
	mcmc_priors=None,
	):
	band_names = lcset_info['band_names']
	class_names = lcset_info['class_names']
	c = class_names[lcobj.y]
	ignored = is_in_column(lcobj_name, sne_specials_df, 'fit_ignored')

	### generate curves
	gc_kwargs = {
		'ignored':ignored,
		'mcmc_priors':mcmc_priors,
	}
	cmethod = '-'.join(method.split('-')[:-1])
	sne_generator = get_syn_sne_generator(cmethod)(lcobj, class_names, band_names, obse_sampler_bdict, uses_estw, **gc_kwargs)
	new_lcobjs, new_smooth_lcojbs, trace_bdict, segs = sne_generator.sample_curves(synthetic_samples_per_curve)

	### save file
	to_save = {
		'lcobj_name':lcobj_name,
		'lcobj':lcobj,
		'band_names':band_names,
		'c':c,
		'new_lcobjs':[new_lcobj.copy().reset_day_offset_serial() for new_lcobj in new_lcobjs],
		'trace_bdict':trace_bdict,
		'segs':segs,
		'ignored':ignored,
		'synthetic_samples_per_curve':synthetic_samples_per_curve,
	}
	save_filedir = f'{save_rootdir}/{method}/{lcobj_name}.ssne'
	save_pickle(save_filedir, to_save, verbose=0) # save error file

	### save images
	need_to_save_images = True
	#need_to_save_images = not 'spm-mle' in method
	if need_to_save_images:
		save_filedirs = [f'{save_rootdir}/__sne-figs/{c}/{method}/{lcobj_name}.png']
		if is_in_column(lcobj_name, sne_specials_df, 'vis'):
			#save_filedirs += [f'{save_rootdir}/__figs__/__vis__/{method}/{lcobj_name}.png']
			pass

		plot_kwargs = {
			'trace_bdict':trace_bdict,
			'save_filedir':save_filedirs,
		}
		plot_synthetic_samples(lcobj_name, lcobj, lcset_name, lcset_info, method, new_lcobjs, new_smooth_lcojbs, **plot_kwargs)
	return

def generate_synthetic_dataset(lcdataset, lcset_name, obse_sampler_bdict, uses_estw, save_rootdir,
	method=None,
	synthetic_samples_per_curve:float=4,
	sne_specials_df=None,
	mcmc_priors=None,
	backend=C_.JOBLIB_BACKEND,
	n_jobs=C_.N_JOBS,
	chunk_size=C_.CHUNK_SIZE,
	):
	lcset = lcdataset[lcset_name]
	lcobj_names = [n for n in lcset.get_lcobj_names() if not filedir_exists(f'{save_rootdir}/{method}/{n}.ssne')]
	chunks = get_list_chunks(lcobj_names, chunk_size)
	bar = ProgressBar(len(chunks))
	for kc,chunk in enumerate(chunks):
		bar(f'lcset_name={lcset_name} - chunck={kc} - chunk_size={chunk_size} - method={method} - chunk={chunk}')
		jobs = []

		for lcobj_name in chunk:
			jobs.append(delayed(generate_synthetic_samples)(lcobj_name, lcset.get_copy(lcobj_name), lcset_name, lcset.get_info(), obse_sampler_bdict, uses_estw, save_rootdir,
				method,
				synthetic_samples_per_curve,
				sne_specials_df,
				mcmc_priors,
				))

		results = Parallel(n_jobs=n_jobs, backend=backend)(jobs)
	bar.done()