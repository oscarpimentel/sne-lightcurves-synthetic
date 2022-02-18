from __future__ import print_function
from __future__ import division
from . import _C
import cProfile

import numpy as np
from fuzzytools.progress_bars import ProgressBar
from fuzzytools.files import filedir_exists, PFile
from fuzzytools.matplotlib.utils import save_fig
from .plots.lc import plot_synthetic_samples
from .generators import ssne_generators as ssneg
from fuzzytools.multiprocessing import get_joblib_config_batches
from joblib import Parallel, delayed

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
	cmethod = '-'.join(method.split('-')[:-1])
	sne_generator = get_syn_sne_generator(cmethod)(lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		ignored=ignored,
		mcmc_priors=mcmc_priors,
		)
	new_lcobjs_with_offset, new_smooth_lcojbs, trace_bdict, segs = sne_generator.sample_curves(synthetic_samples_per_curve)
	new_lcobjs = [new_lcobj.copy().reset_day_offset_serial() for new_lcobj in new_lcobjs_with_offset]

	d = {
		'lcobj_name':lcobj_name,
		'lcobj':lcobj,
		'band_names':band_names,
		'c':c,
		'new_lcobjs_with_offset':new_lcobjs_with_offset,
		'new_smooth_lcojbs':new_smooth_lcojbs,
		'new_lcobjs':new_lcobjs,
		'trace_bdict':trace_bdict,
		'segs':segs,
		'ignored':ignored,
		'synthetic_samples_per_curve':synthetic_samples_per_curve,
		'img_filedir':f'{figs_save_rootdir}/{c}/{lcobj_name}.pdf',
		}
	pfile = PFile(f'{ssne_save_rootdir}/{lcobj_name}.ssne', d)
	return pfile

def generate_synthetic_dataset(lcset_name, lcset, obse_sampler_bdict, uses_estw, ssne_save_rootdir, figs_save_rootdir,
	method=None,
	synthetic_samples_per_curve:float=4,
	sne_specials_df=None,
	mcmc_priors=None,
	backend='multiprocessing',
	):
	lcset_info = lcset.get_info()
	lcobj_names = lcset.get_lcobj_names()
	lcobj_names = [lcobj_name for lcobj_name in lcset.get_lcobj_names() if not filedir_exists(f'{ssne_save_rootdir}/{lcobj_name}.ssne')]
	# lcobj_names = ['ZTF19abpnsck', 'ZTF19aazdsgo']
	batches, n_jobs = get_joblib_config_batches(lcobj_names, backend)
	bar = ProgressBar(len(batches))
	for kc,batch in enumerate(batches):
		bar(f'method={method}; lcset_name={lcset_name}; samples={synthetic_samples_per_curve}; batch={batch}({len(batch)}#)')
		jobs = []
		for lcobj_name in batch:
			if lcobj_name in lcset.get_lcobj_names():
				jobs.append(delayed(generate_synthetic_samples)(
					lcobj_name,
					lcset[lcobj_name],
					lcset_name,
					lcset_info,
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
		for pfile in results:
			d = pfile.file
			pfile.save()
			fig, axs = plot_synthetic_samples(d['lcobj_name'], d['lcobj'], lcset_name, lcset_info, method, d['new_lcobjs_with_offset'], d['new_smooth_lcojbs'],
				trace_bdict=d['trace_bdict'],
				)
			fig.tight_layout()
			save_fig(fig, d['img_filedir'])

	bar.done()