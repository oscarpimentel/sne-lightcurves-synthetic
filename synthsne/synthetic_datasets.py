from __future__ import print_function
from __future__ import division
from . import C_
import cProfile

import numpy as np
from flamingchoripan.progress_bars import ProgressBar
from flamingchoripan.files import filedir_exists, PFile
from flamingchoripan.cuteplots.utils import IFile
from .plots.lc import plot_synthetic_samples
from flamingchoripan.lists import get_list_chunks
from joblib import Parallel, delayed
from .generators import ssne_generators as ssneg

###################################################################################################################################################

def get_syn_sne_generator(method):
	if method=='linear':
		return ssneg.SynSNeGeneratorLinear
	if method=='bspline':
		return ssneg.SynSNeGeneratorBSpline
	if method=='spm-mle':
		return ssneg.SynSNeGeneratorMLE
	if method=='spm-mcmc':
		return ssneg.SynSNeGeneratorMCMC
	raise Exception(f'no method {method}')

def is_in_column(lcobj_name, sne_specials_df, column):
	if sne_specials_df is None:
		return False
	return lcobj_name in list(sne_specials_df[column].values)

###################################################################################################################################################

def generate_synthetic_samples(lcobj_name, lcobj, lcset_name, lcset_info, obse_sampler_bdict, uses_estw, ssne_save_rootdir, figs_save_rootdir,
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

	### ssne save file
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
	pfile = PFile(f'{ssne_save_rootdir}/{lcobj_name}.ssne', to_save)

	### save images
	bypass_img_saving = False
	#bypass_img_saving = 'spm-mle' in method
	if bypass_img_saving:
		img_filedir = None
		fig = None
	else:
		img_filedir = f'{figs_save_rootdir}/{c}/{lcobj_name}.png'
		plot_kwargs = {
			'trace_bdict':trace_bdict,
			}
		fig, axs = plot_synthetic_samples(lcobj_name, lcobj, lcset_name, lcset_info, method, new_lcobjs, new_smooth_lcojbs, **plot_kwargs)
	ifile = IFile(img_filedir, fig)
	return pfile, ifile

def generate_synthetic_dataset(lcset_name, lcset, obse_sampler_bdict, uses_estw, ssne_save_rootdir, figs_save_rootdir,
	method=None,
	synthetic_samples_per_curve:float=4,
	sne_specials_df=None,
	mcmc_priors=None,
	backend=C_.JOBLIB_BACKEND,
	n_jobs=C_.N_JOBS,
	chunk_size=C_.CHUNK_SIZE,
	):
	lcobj_names = [lcobj_name for lcobj_name in lcset.get_lcobj_names() if not filedir_exists(f'{ssne_save_rootdir}/{lcobj_name}.ssne')]
	#lcobj_names = [lcobj_name for lcobj_name in lcset.get_lcobj_names()]
	#lcobj_names = ['ZTF20aasfhia', 'ZTF19aassqix', 'ZTF19aauivtj','ZTF19aczeomw', 'ZTF19abfibel', 'ZTF19acjwdnu', 'ZTF19adbryab', 'ZTF18abeajml', 'ZTF18aaxkfos', 'ZTF19aarnqys', 'ZTF19aailcgs']
	#lcobj_names = ['ZTF19aailcgs']
	chunks = get_list_chunks(lcobj_names, chunk_size)
	bar = ProgressBar(len(chunks))
	for kc,chunk in enumerate(chunks):
		bar(f'method={method}  - lcset_name={lcset_name} - samples={synthetic_samples_per_curve} - chunk={chunk}')
		jobs = []
		for lcobj_name in chunk:
			if lcobj_name in lcset.get_lcobj_names():
			#if 1:
				jobs.append(delayed(generate_synthetic_samples)(
					lcobj_name,
					lcset[lcobj_name],
					lcset_name,
					lcset.get_info(),
					obse_sampler_bdict,
					uses_estw,
					ssne_save_rootdir,
					figs_save_rootdir,
					method,
					synthetic_samples_per_curve,
					sne_specials_df,
					mcmc_priors,
					))
		results = Parallel(n_jobs=n_jobs, backend=backend)(jobs)
		for pfile, ifile in results:
			pfile.save()
			ifile.save()
	bar.done()