from __future__ import print_function
from __future__ import division
from . import C_

import fuzzytools.files as fcfiles
from fuzzytools.datascience.xerror import XError
from fuzzytools.datascience.ranks import TopRank
from fuzzytools.dataframes import DFBuilder
from fuzzytools.lists import flat_list
import numpy as np
import pandas as pd
from nested_dict import nested_dict

###################################################################################################################################################

def empty_roodir(rootdir):
	return len(fcfiles.get_filedirs(rootdir))==0

def get_band_names(rootdir):
	filedirs = fcfiles.get_filedirs(rootdir)
	filedir = filedirs[0]
	return fcfiles.load_pickle(filedir)['band_names']

def get_classes(rootdir):
	classes = []
	filedirs = fcfiles.get_filedirs(rootdir)
	for filedir in filedirs:
		fdict = fcfiles.load_pickle(filedir)
		c = fdict['c']
		if not c in classes:
			classes.append(c)
	return classes

def get_spm_args(rootdir, spm_p, b, c):
	files = fcfiles.gather_files(rootdir, fext=None)
	spm_args = []
	for f in files:
		if not c==f()['c']:
			continue

		sne_models = f()['trace_bdict'][b].sne_models
		for sne_model in sne_models:
			if not sne_model is None:
				if not sne_model.spm_args is None:
					spm_args += [sne_model.spm_args[spm_p]]

	return spm_args

###################################################################################################################################################

def get_any_incorrects_fittings(rootdir, kf, lcset_name):
	files, files_ids = fcfiles.gather_files_by_kfold(rootdir, kf, lcset_name)
	lcobj_names = []
	for f in files:
		if any([new_lcobj.any_real() for new_lcobj in f()['new_lcobjs']]):
			lcobj_names.append(f()['lcobj_name'])
	return lcobj_names

def get_all_incorrects_fittings(rootdir, kf, lcset_name):
	files, files_ids = fcfiles.gather_files_by_kfold(rootdir, kf, lcset_name)
	lcobj_names = []
	for f in files:
		if all([new_lcobj.all_real() for new_lcobj in f()['new_lcobjs']]):
			lcobj_names.append(f()['lcobj_name'])
	return lcobj_names

def get_perf_times(rootdir, kf, lcset_name):
	files, files_ids = fcfiles.gather_files_by_kfold(rootdir, kf, lcset_name)
	times = []
	for f in files:
		if all([new_lcobj.all_synthetic() for new_lcobj in f()['new_lcobjs']]):
			times.append(f()['segs'])
	return XError(times)

def get_info_dict(rootdir, methods, cfilename, kf, lcset_name,
	band_names=['g', 'r'],
	):
	info_df = DFBuilder()

	### all info
	d = {}
	for method in methods:
		_rootdir = f'{rootdir}/{method}/{cfilename}'
		files, files_ids = fcfiles.gather_files_by_kfold(_rootdir, kf, lcset_name)
		trace_time = [f()['segs'] for f in files]
		d[method] = XError(trace_time)

	info_df.append(f'metric=trace-time [segs]~band=.', d)

	### per band info
	for kb,b in enumerate(band_names):
		d = nested_dict()
		for method in methods:
			_rootdir = f'{rootdir}/{method}/{cfilename}'
			files, files_ids = fcfiles.gather_files_by_kfold(_rootdir, kf, lcset_name)
			traces = [f()['trace_bdict'][b] for f in files]
			trace_errors = flat_list([t.get_valid_errors() for t in traces])
			trace_errors_xe = XError(np.log(np.array(trace_errors)+C_.EPS))
			d['error'][method] = trace_errors_xe
			d['success'][method] = len(trace_errors)/sum([len(t) for t in traces])*100

		d = d.to_dict()
		info_df.append(f'metric=fit-log-error~band={b}', d['error'])
		info_df.append(f'metric=fits-success [%]~band={b}', d['success'])
	
	return info_df.get_df()

def get_ranks(rootdir, kf, lcset_name,
	band_names=['g', 'r'],
	):
	files, files_ids = fcfiles.gather_files_by_kfold(rootdir, kf, lcset_name)
	rank_bdict = {b:TopRank(f'band={b}') for b in band_names}
	for f,fid in zip(files, files_ids):
		lcobj_name = f()['lcobj_name']
		for b in band_names:
			errors = f()['trace_bdict'][b].get_valid_errors()
			if len(errors)==0:
				continue

			xe = XError(errors)
			rank_bdict[b].append(fid, xe.mean)
	
	for b in band_names:
		rank_bdict[b].calcule()
	return rank_bdict