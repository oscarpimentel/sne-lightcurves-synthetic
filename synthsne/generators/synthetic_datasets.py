from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from flamingchoripan.progress_bars import ProgressBar
from flamingchoripan.files import save_pickle, check_filedir_exists
from .synthetic_curves import get_syn_sne_generator
from ..plots.lc import plot_synthetic_samples
from flamingchoripan.lists import get_list_chunks
from joblib import Parallel, delayed

###################################################################################################################################################

def is_in_column(lcobj_name, sne_specials_df, column):
	if sne_specials_df is None:
		return False
	return lcobj_name in list(sne_specials_df[column].values)

def generate_synthetic_samples(lcobj_name, lcset, lcset_name, obse_sampler_bdict, length_sampler_bdict, uses_estw, save_rootdir,
	method='linear',
	synthetic_samples_per_curve:float=4,
	add_original=True,
	sne_specials_df=None,
	):
	band_names = lcset.band_names
	class_names = lcset.class_names
	lcobj = lcset[lcobj_name]
	c = class_names[lcobj.y]
	ignored = is_in_column(lcobj_name, sne_specials_df, 'fit_ignored')

	### generate curves
	gc_kwargs = {
		'ignored':ignored,
	}
	cmethod = '-'.join(method.split('-')[:-1])
	sne_generator = get_syn_sne_generator(cmethod)(lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict, uses_estw, **gc_kwargs)
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
	}
	save_filedir = f'{save_rootdir}/{method}/{lcobj_name}.synsne'
	save_pickle(save_filedir, to_save, verbose=0) # save error file

	### save images
	save_filedirs = [f'{save_rootdir}/__figs__/{c}/{method}/{lcobj_name}.png']
	if is_in_column(lcobj_name, sne_specials_df, 'vis'):
		save_filedirs += [f'{save_rootdir}/__figs__/__vis__/{method}/{lcobj_name}.png']

	plot_kwargs = {
		'trace_bdict':trace_bdict,
		'save_filedir':save_filedirs,
	}
	plot_synthetic_samples(lcset, lcset_name, method, lcobj_name, new_lcobjs, new_smooth_lcojbs, **plot_kwargs)
	return

def generate_synthetic_dataset(lcdataset, lcset_name, obse_sampler_bdict, length_sampler_bdict, uses_estw, save_rootdir,
	method='linear',
	synthetic_samples_per_curve:float=4,
	add_original=True,
	sne_specials_df=None,
	n_jobs=C_.N_JOBS,
	chunk_size=C_.CHUNK_SIZE,
	backend=None,
	#backend='threading', # explodes with pymc3
	remove_lock_dir=True,
	):
	lcset = lcdataset[lcset_name]
	lcobj_names = [n for n in lcset.get_lcobj_names() if not check_filedir_exists(f'{save_rootdir}/{method}/{n}.synsne')]
	chunks = get_list_chunks(lcobj_names, chunk_size)
	bar = ProgressBar(len(chunks))
	
	for kc,chunk in enumerate(chunks):
		bar(f'lcset_name: {lcset_name} - chunck: {kc} - chunk_size: {chunk_size} - method: {method} - chunk:{chunk}')
		jobs = []

		for lcobj_name in chunk:
			jobs.append(delayed(generate_synthetic_samples)(lcobj_name, lcset, lcset_name, obse_sampler_bdict, length_sampler_bdict, uses_estw, save_rootdir,
				method,
				synthetic_samples_per_curve,
				add_original,
				sne_specials_df,
				))

		results = Parallel(n_jobs=n_jobs, backend=backend)(jobs)
	bar.done()