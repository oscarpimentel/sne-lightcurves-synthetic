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
	parser.add_argument('-method',  type=str, default='all', help='method')
	parser.add_argument('-set',  type=str, default='train', help='set')
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
	from synthsne import C_

	def load_lcdataset(filename):
		assert filename.split('.')[-1]==C_.EXT_SPLIT_LIGHTCURVE
		return load_pickle(filename)

	filedir = '../../surveys-save/alerceZTFv7.1/survey-alerceZTFv7.1_bands-gr_mode-onlySNe.splcds'

	filedict = get_dict_from_filedir(filedir)
	root_folder = filedict['*rootdir*']
	cfilename = filedict['*cfilename*']
	survey = filedict['survey']
	lcdataset = load_lcdataset(filedir)
	print(lcdataset['raw'].keys())
	print(lcdataset['raw'].get_random_lcobj(False).keys())
	print(lcdataset)

	###################################################################################################################################################
	from synthsne.generators.synthetic_datasets import generate_synthetic_dataset
	import pandas as pd
	from synthsne import C_
	import flamingchoripan.files as ff
	from flamingchoripan.progress_bars import ProgressBar
	from flamingchoripan.files import load_pickle, save_pickle
	
	methods = main_args.method
	if methods=='all':
		methods = ['linear', 'curvefit', 'bspline', 'uniformprior', 'mcmc']

	if isinstance(methods, str):
		methods = [methods]

	lcset_name = main_args.set
	for method in methods:
		save_rootdir = f'../save/{survey}/{cfilename}/{lcset_name}'
		sd_kwargs = {
			'synthetic_samples_per_curve':64,
			'method':method,
			'sne_specials_df':pd.read_csv(f'../data/{survey}/sne_specials.csv'),
		}
		samplers = load_pickle(f'{save_rootdir}/samplers.{C_.EXT_SAMPLER}')
		obse_sampler_bdict = samplers['obse_sampler_bdict']
		length_sampler_bdict = samplers['length_sampler_bdict']
		generate_synthetic_dataset(lcdataset, lcset_name, obse_sampler_bdict, length_sampler_bdict, save_rootdir, **sd_kwargs)

		### export
		lcset = lcdataset[lcset_name]
		synth_rootdir = f'../save/{survey}/{cfilename}/{lcset_name}/{method}'
		print('synth_rootdir:', synth_rootdir)
		synth_lcset = lcset.copy({})
		filedirs = ff.get_filedirs(synth_rootdir, fext='synsne')
		bar = ProgressBar(len(filedirs))
		for filedir in filedirs:
			d = ff.load_pickle(filedir, verbose=0)
			lcobj_name = d['lcobj_name']
			bar(f'{lcobj_name}')
			lcobj = d['lcobj']
			synth_lcset.set_lcobj(f'{lcobj_name}.0', lcobj) # set orinal anyways
			
			has_corrects_samples = d['has_corrects_samples']
			ignored = d['ignored']
			if has_corrects_samples and not ignored:
				for k,new_lcobj in enumerate(d['new_lcobjs']):
					synth_lcset.set_lcobj(f'{lcobj_name}.{k+1}', new_lcobj)

		bar.done()
		new_lcset_name = f'{lcset_name}.{method}'
		lcdataset.set_lcset(new_lcset_name, synth_lcset)

		save_rootdir = f'{root_folder}'
		save_filedir = f'{save_rootdir}/{cfilename}_method-{method}.{C_.EXT_SPLIT_LIGHTCURVE}'
		save_pickle(save_filedir, lcdataset)
		lcdataset.del_lcset(new_lcset_name)