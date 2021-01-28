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
	main_args = parser.parse_args()
	print_big_bar()

	###################################################################################################################################################
	from flamingchoripan.files import search_for_filedirs
	from synthsne import C_

	root_folder = '../../surveys-save'
	filedirs = search_for_filedirs(root_folder, fext=C_.EXT_SPLIT_LIGHTCURVE)

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle
	from flamingchoripan.files import get_dict_from_filedir

	filedir = '../../surveys-save/alerceZTFv7.1/survey=alerceZTFv7.1°bands=gr°mode=onlySNe.splcds'
	filedict = get_dict_from_filedir(filedir)
	root_folder = filedict['*rootdir*']
	cfilename = filedict['*cfilename*']
	survey = filedict['survey']
	lcdataset = load_pickle(filedir)
	print(lcdataset)

	###################################################################################################################################################
	import numpy as np
	import flamingchoripan.files as ff
	from flamingchoripan.progress_bars import ProgressBar
	from flamingchoripan.files import load_pickle, save_pickle
	from synthsne import C_

	methods = main_args.method
	methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mle-estw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if methods=='.' else methods
	methods = [methods] if isinstance(methods, str) else methods

	for method in methods:
		new_lcdataset = lcdataset.copy()
		lcset_names = ['train', 'val']
		lcset_names = ['train']
		for lcset_name in lcset_names:
			lcset = new_lcdataset[lcset_name]
			synth_rootdir = f'../save/{survey}/{cfilename}/{lcset_name}/{method}'
			print('synth_rootdir:', synth_rootdir)
			synth_lcset = lcset.copy({})
			filedirs = ff.get_filedirs(synth_rootdir, fext='synsne')
			bar = ProgressBar(len(filedirs))
			for filedir in filedirs:
				d = ff.load_pickle(filedir, verbose=0)
				lcobj_name = d['lcobj_name']
				bar(f'lcset_name: {lcset_name} - lcobj_name: {lcobj_name}')
				#synth_lcset.set_lcobj(f'{lcobj_name}.0', d['lcobj']) # set orinal anyways
				
				for k,new_lcobj in enumerate(d['new_lcobjs']):
					synth_lcset.set_lcobj(f'{lcobj_name}.{k+1}', new_lcobj)

			bar.done()
			new_lcset_name = f'{lcset_name}.{method}'
			new_lcdataset.set_lcset(new_lcset_name, synth_lcset)

		save_rootdir = f'{root_folder}'
		save_filedir = f'{save_rootdir}/{cfilename}°method={method}.{C_.EXT_SPLIT_LIGHTCURVE}'
		save_pickle(save_filedir, new_lcdataset)