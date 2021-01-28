from __future__ import print_function
from __future__ import division
from . import C_

import flamingchoripan.files as ff
from flamingchoripan.datascience.statistics import XError, TopRank
import numpy as np
import pandas as pd

###################################################################################################################################################

def get_filedirs(rootdir, method):
	load_rootdir = f'{rootdir}/{method}'
	filedirs = ff.get_filedirs(load_rootdir, fext='synsne')
	return filedirs

def get_band_names(rootdir, method):
	filedirs = get_filedirs(rootdir, method)
	return ff.load_pickle(filedirs[0], verbose=0)['band_names']

def get_all_incorrects_fittings(rootdir, method):
	filedirs = get_filedirs(rootdir, method)
	obj_names = []
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		if not fdict['has_corrects_samples']:
			obj_names.append(fdict['lcobj_name'])

	return obj_names

def get_perf_times(rootdir, method):
	filedirs = get_filedirs(rootdir, method)
	times = []
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		if fdict['has_corrects_samples']:
			times.append(fdict['segs'])

	return XError(times)

def get_classes(rootdir, method):
	classes = []
	filedirs = get_filedirs(rootdir, method)
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		c = fdict['c']
		if not c in classes:
			classes.append(c)

	return classes

def get_spm_parameters(rootdir, method, b):
	filedirs = get_filedirs(rootdir, method)
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		if fdict['has_corrects_samples']:
			sne_models = fdict['trace_bdict'][b].sne_model_l
			for sne_model in sne_models:
				if not sne_model is None:
					return sne_model.parameters

def get_spm_args(rootdir, method, b):
	filedirs = get_filedirs(rootdir, method)
	spm_args = {p:[] for p in get_spm_parameters(rootdir, method, b)}
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		if fdict['has_corrects_samples']:
			#print(fdict.keys())
			c = fdict['c']
			sne_models = fdict['trace_bdict'][b].sne_model_l
			for sne_model in sne_models:
				if not sne_model is None:
					for p in sne_model.parameters:
						spm_args[p].append({'p':sne_model.spm_args[p], 'c':c})

	return spm_args

def get_ranks(rootdir, method):
	band_names = get_band_names(rootdir, method)
	rank = TopRank('mb-rank')
	rank_bdict = {b:TopRank(f'{b}-rank') for b in band_names}
	filedirs = get_filedirs(rootdir, method)
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		lcobj_name = fdict['lcobj_name']
		xes = []
		for b in band_names:
			errors = fdict['trace_bdict'][b].get_valid_errors()
			if len(errors)>0:
				xe = XError(errors, 0)
				rank_bdict[b].add(lcobj_name, xe.mean)
				xes.append(xe)
				
		if len(xes)>0:
			rank.add(lcobj_name, np.mean([xe.mean for xe in xes]))
			
	return rank, rank_bdict, band_names

def get_info_dict(rootdir, methods):
	band_names = get_band_names(rootdir, methods[0])
	info_dict = {}
	# $\chi_{\sigma_x^2}$
	for method in methods:
		method_k = f'method={method}'
		info_dict[method_k] = {
			'trace-time [segs]':[],
			'mb-fit-log-error':[],
			'mb-fits-n':0,
			'mb-n':0,
			'mb-fits [%]':None,
		}
		for b in band_names:
			info_dict[method_k].update({
				f'{b}-fit-log-error':[],
				f'{b}-fits-n':0,
				f'{b}-n':0,
				f'{b}-fits [%]':None,
			})

	for method in methods:
		method_k = f'method={method}'
		filedirs = get_filedirs(rootdir, method)
		for filedir in filedirs:
			fdict = ff.load_pickle(filedir, verbose=0)
			lcobj_name = fdict['lcobj_name']
			segs = fdict['segs']
			for b in band_names:
				trace = fdict['trace_bdict'][b]
				errors = trace.get_valid_errors()
				info_dict[method_k][f'{b}-fit-log-error'] += errors
				info_dict[method_k][f'{b}-fits-n'] += len(errors)
				info_dict[method_k][f'{b}-n'] += len(trace)

				### mb
				info_dict[method_k][f'trace-time [segs]'] += [segs]
				info_dict[method_k][f'mb-fit-log-error'] += errors
				info_dict[method_k][f'mb-fits-n'] += len(errors)
				info_dict[method_k][f'mb-n'] += len(trace)

	for method in methods:
		method_k = f'method={method}'
		info_dict[method_k]['trace-time [segs]'] = XError(info_dict[method_k]['trace-time [segs]'])
		info_dict[method_k]['mb-fit-log-error'] = XError(np.log(info_dict[method_k]['mb-fit-log-error']))
		info_dict[method_k]['mb-fits [%]'] = info_dict[method_k].get('mb-fits-n')/info_dict[method_k].pop('mb-n')*100 # get, pop
		for b in band_names:
			info_dict[method_k][f'{b}-fit-log-error'] = XError(np.log(info_dict[method_k][f'{b}-fit-log-error']))
			info_dict[method_k][f'{b}-fits [%]'] = info_dict[method_k].get(f'{b}-fits-n')/info_dict[method_k].pop(f'{b}-n')*100

	info_df = pd.DataFrame.from_dict(info_dict, orient='index').reindex(list(info_dict.keys()))
	info_df = info_df.sort_values(by=['trace-time [segs]'])
	return info_df