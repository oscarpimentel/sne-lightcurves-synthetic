from __future__ import print_function
from __future__ import division
from . import C_

import flamingchoripan.files as ff
from flamingchoripan.datascience.statistics import XError, TopRank
import numpy as np

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