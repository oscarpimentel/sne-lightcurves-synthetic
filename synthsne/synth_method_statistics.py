from __future__ import print_function
from __future__ import division
from . import C_

import flamingchoripan.files as ff
from flamingchoripan.datascience.statistics import XError, TopRank
import numpy as np
import pandas as pd
import math

###################################################################################################################################################

'''
def get_filedirs(rootdir, method,
	fext=None, # synsne
	):
	load_rootdir = f'{rootdir}/{method}'
	filedirs = ff.get_filedirs(load_rootdir, fext=fext)
	return filedirs
'''

def get_band_names(rootdir, method):
	filedirs = get_filedirs(rootdir, method)
	return ff.load_pickle(filedirs[0], verbose=0)['band_names']

def get_classes(rootdir, method):
	classes = []
	filedirs = get_filedirs(rootdir, method)
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		c = fdict['c']
		if not c in classes:
			classes.append(c)
	return classes

def get_any_incorrects_fittings(rootdir, method):
	filedirs = get_filedirs(rootdir, method)
	obj_names = []
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		if any([new_lcobj.any_real() for new_lcobj in fdict['new_lcobjs']]):
			obj_names.append(fdict['lcobj_name'])

	return obj_names

def get_all_incorrects_fittings(rootdir, method):
	filedirs = get_filedirs(rootdir, method)
	obj_names = []
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		if all([new_lcobj.all_real() for new_lcobj in fdict['new_lcobjs']]):
			obj_names.append(fdict['lcobj_name'])

	return obj_names

def get_perf_times(rootdir, method):
	filedirs = get_filedirs(rootdir, method)
	times = []
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		if all([new_lcobj.all_synthetic() for new_lcobj in fdict['new_lcobjs']]):
			times.append(fdict['segs'])

	return XError(times)

def get_spm_parameters(rootdir, method, b):
	filedirs = get_filedirs(rootdir, method)
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		#if fdict['has_corrects_samples']:
		sne_models = fdict['trace_bdict'][b].sne_model_l
		for sne_model in sne_models:
			if not sne_model is None:
				return sne_model.parameters

def get_spm_args(rootdir, spm_p, b, c):
	filedirs = ff.get_filedirs(rootdir, fext=None)
	spm_args = []
	for filedir in filedirs:
		fdict = ff.load_pickle(filedir, verbose=0)
		#if fdict['has_corrects_samples']:
		#print(fdict.keys())
		if not c==fdict['c']:
			continue

		sne_models = fdict['trace_bdict'][b].sne_model_l
		for sne_model in sne_models:
			if not sne_model is None: # filter incorrect fits
				#for p in sne_model.parameters:
				# {'p':sne_model.spm_args[p], 'c':c})
				#print(p)
				spm_args += [sne_model.spm_args[spm_p]]

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
	info_dict.update({
		'trace-time [segs]':{},
		'mb-fit-log-error':{},
		'mb-fits-n':{},
		'mb-n':{},
		'mb-fits [%]':{},
	})
	for b in band_names:
		info_dict.update({
			f'{b}-fit-log-error':{},
			f'{b}-fits-n':{},
			f'{b}-n':{},
			f'{b}-fits [%]':{},
		})
	for method in methods:
		filedirs = get_filedirs(rootdir, method)
		for filedir in filedirs:
			fdict = ff.load_pickle(filedir, verbose=0)
			lcobj_name = fdict['lcobj_name']
			segs = fdict['segs']

			for b in band_names:
				trace = fdict['trace_bdict'][b]
				errors = trace.get_valid_errors()

				### b
				try:
					info_dict[f'{b}-fit-log-error'][method] += [math.log(e+1e-10) for e in errors]
				except KeyError:
					info_dict[f'{b}-fit-log-error'][method] = []

				try:
					info_dict[f'{b}-fits-n'][method].append(len(errors))
				except KeyError:
					info_dict[f'{b}-fits-n'][method] = []

				try:
					info_dict[f'{b}-n'][method].append(len(trace))
				except KeyError:
					info_dict[f'{b}-n'][method] = []

				### mb
				try:
					info_dict['trace-time [segs]'][method].append(segs)
				except KeyError:
					info_dict['trace-time [segs]'][method] = []

				try:
					info_dict['mb-fit-log-error'][method] += [math.log(e+1e-10) for e in errors]
				except KeyError:
					info_dict['mb-fit-log-error'][method] = []

				try:
					info_dict['mb-fits-n'][method].append(len(errors))
				except KeyError:
					info_dict['mb-fits-n'][method] = []

				try:
					info_dict['mb-n'][method].append(len(trace))
				except KeyError:
					info_dict['mb-n'][method] = []

	for method in methods:
		info_dict['mb-fits-n'][method] = sum(info_dict['mb-fits-n'][method])
		info_dict['mb-n'][method] = sum(info_dict['mb-n'][method])
		info_dict['mb-fits [%]'][method] = info_dict['mb-fits-n'][method]/info_dict['mb-n'][method]*100
		for b in band_names:
			info_dict[f'{b}-fits-n'][method] = sum(info_dict[f'{b}-fits-n'][method])
			info_dict[f'{b}-n'][method] = sum(info_dict[f'{b}-n'][method])
			info_dict[f'{b}-fits [%]'][method] = info_dict[f'{b}-fits-n'][method]/info_dict[f'{b}-n'][method]*100

	info_dict = {f'metric={i}':info_dict[i] for i in info_dict.keys() if not '-n' in i}
	info_df = pd.DataFrame.from_dict(info_dict, orient='index').reindex(info_dict.keys())
	for c in info_df.columns:
		info_df[c].values[:] = [XError(v) if isinstance(v, list) else v for v in info_df[c].values[:]] # make xerror from list

	return info_df