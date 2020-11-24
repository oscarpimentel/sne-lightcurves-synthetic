from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from flamingchoripan.progress_bars import ProgressBar
from flamingchoripan.files import save_pickle
from .synthetic_curves import get_syn_sne_generator
from ..plots.lc import plot_synthetic_samples
from flamingchoripan.strings import get_list_chunks
from joblib import Parallel, delayed

###################################################################################################################################################

def is_in_column(lcobj_name, sne_specials_df, column):
	if sne_specials_df is None:
		return False
	return lcobj_name in list(sne_specials_df[column].values)

def generate_synthetic_samples(lcobj_name, lcdataset, lcset_name, obse_sampler_bdict, length_sampler_bdict, save_rootdir,
	method='curve_fit',
	synthetic_samples_per_curve:float=4,
	add_original=True,
	sne_specials_df=None,
	):
	lcset = lcdataset[lcset_name]
	band_names = lcset.band_names
	class_names = lcset.class_names
	lcobj = lcset[lcobj_name]
	c = class_names[lcobj.y]

	### generate curves
	sne_generator = get_syn_sne_generator(method)(lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict)
	new_lcobjs, new_smooth_lcojbs, trace_bdict, has_corrects_samples = sne_generator.sample_curves(synthetic_samples_per_curve, return_has_corrects_samples=True)

	### save file
	ignored = is_in_column(lcobj_name, sne_specials_df, 'fit_ignored')
	outlier = is_in_column(lcobj_name, sne_specials_df, 'outliers')
	method_folder = f'{method}_outliers' if outlier else method

	to_save = {
		'lcobj_name':lcobj_name,
		'lcobj':lcobj,
		'band_names':band_names,
		'c':c,
		'new_lcobjs':new_lcobjs,
		'trace_bdict':trace_bdict,
		'has_corrects_samples':has_corrects_samples,
		'ignored':ignored,
	}
	save_filedir = f'{save_rootdir}/{method_folder}/{lcobj_name}.synsne'
	save_pickle(save_filedir, to_save, verbose=0) # save error file

	### save images
	save_filedirs = [f'{save_rootdir}/{method_folder}/{lcobj_name}.png', f'{save_rootdir}/{method}_{c}/{lcobj_name}.png']
	if is_in_column(lcobj_name, sne_specials_df, 'vis'):
		save_filedirs += [f'{save_rootdir}/{method}_vis/{lcobj_name}.png']

	plot_kwargs = {
		'trace_bdict':trace_bdict,
		'save_filedir':save_filedirs,
	}
	plot_synthetic_samples(lcdataset, lcset_name, method, lcobj_name, new_lcobjs, new_smooth_lcojbs, **plot_kwargs)

def generate_synthetic_dataset(lcdataset, lcset_name, obse_sampler_bdict, length_sampler_bdict, save_rootdir,
	method='curve_fit',
	synthetic_samples_per_curve:float=4,
	add_original=True,
	sne_specials_df=None,
	n_jobs=C_.N_JOBS,
	chunk_size=C_.CHUNK_SIZE,
	):
	lcset = lcdataset[lcset_name]
	chunks = get_list_chunks(lcset.get_lcobj_names(), chunk_size)
	bar = ProgressBar(len(chunks))
	for kc,chunk in enumerate(chunks):
		bar(f'lcset_name: {lcset_name} - chunck: {kc} - chunk_size: {chunk_size} - objs: {chunk}')
		jobs = []
		for lcobj_name in chunk:
			jobs.append(delayed(generate_synthetic_samples)(lcobj_name, lcdataset, lcset_name, obse_sampler_bdict, length_sampler_bdict, save_rootdir,
				method,
				synthetic_samples_per_curve,
				add_original,
				sne_specials_df,
				))
		results = Parallel(n_jobs=n_jobs)(jobs)

	bar.done()