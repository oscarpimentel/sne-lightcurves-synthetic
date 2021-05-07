#!/usr/bin/env python3
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../flaming-choripan') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module

if __name__== '__main__':
	### parser arguments
	import argparse
	from flamingchoripan.prints import print_big_bar

	parser = argparse.ArgumentParser('usage description')
	parser.add_argument('-method',  type=str, default='.', help='method')
	parser.add_argument('-kf',  type=str, default='.', help='kf')
	parser.add_argument('-setn',  type=str, default='.', help='kf')
	main_args = parser.parse_args()
	print_big_bar()

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle, get_dict_from_filedir

	filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe.splcds'
	filedict = get_dict_from_filedir(filedir)
	rootdir = filedict['_rootdir']
	cfilename = filedict['_cfilename']
	survey = filedict['survey']
	lcdataset = load_pickle(filedir)
	print(lcdataset)

	###################################################################################################################################################
	import numpy as np
	import flamingchoripan.files as fcfiles
	from flamingchoripan.progress_bars import ProgressBar
	from flamingchoripan.files import load_pickle, save_pickle
	from synthsne import C_

	kfs = lcdataset.kfolds if main_args.kf=='.' else main_args.kf
	kfs = [kfs] if isinstance(kfs, str) else kfs
	methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mle-estw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if main_args.method=='.' else main_args.method
	methods = [methods] if isinstance(methods, str) else methods
	setns = [str(setn) for setn in ['train']] if main_args.setn=='.' else main_args.setn
	setns = [setns] if isinstance(setns, str) else setns

	for method in methods:
		new_lcdataset = lcdataset.copy() # copy with all original lcsets
		for setn in setns:
			for kf in kfs:
				lcset_name = f'{kf}@{setn}'
				lcset = new_lcdataset[lcset_name]
				synth_rootdir = f'../save/ssne/{method}/{cfilename}/{lcset_name}'
				print('synth_rootdir=', synth_rootdir)
				synth_lcset = lcset.copy({}) # copy
				filedirs = fcfiles.get_filedirs(synth_rootdir, fext='ssne')
				bar = ProgressBar(len(filedirs))

				for filedir in filedirs:
					d = fcfiles.load_pickle(filedir)
					lcobj_name = d['lcobj_name']
					bar(f'lcset_name={lcset_name} - lcobj_name={lcobj_name}')

					for k,new_lcobj in enumerate(d['new_lcobjs']):
						synth_lcset.set_lcobj(f'{lcobj_name}.{k+1}', new_lcobj)

				bar.done()
				new_lcset_name = f'{lcset_name}.{method}'
				new_lcdataset.set_lcset(new_lcset_name, synth_lcset)

		save_rootdir = f'{rootdir}'
		save_filedir = f'{save_rootdir}/{cfilename}~method={method}.{C_.EXT_SPLIT_LIGHTCURVE}'
		save_pickle(save_filedir, new_lcdataset)