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
	from flamingchoripan.files import load_pickle, save_pickle, get_dict_from_filedir
	from synthsne import C_

	def load_lcdataset(filename):
		assert filename.split('.')[-1]==C_.EXT_SPLIT_LIGHTCURVE
		return load_pickle(filename)

	filedir = '../../surveys-save/alerceZTFv7.1/survey=alerceZTFv7.1°bands=gr°mode=onlySNe.splcds'

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
		methods = ['linear', 'curvefit', 'bspline', 'mcmc']

	if isinstance(methods, str):
		methods = [methods]

	lcset_name = main_args.set
	for method in methods:
		save_rootdir = f'../save/{survey}/{cfilename}/{lcset_name}'
		sd_kwargs = {
			'synthetic_samples_per_curve':32, # 16 32 64
			'method':method,
			'sne_specials_df':pd.read_csv(f'../data/{survey}/sne_specials.csv'),
		}
		samplers = load_pickle(f'{save_rootdir}/samplers.{C_.EXT_SAMPLER}')
		obse_sampler_bdict = samplers['obse_sampler_bdict']
		length_sampler_bdict = samplers['length_sampler_bdict']
		generate_synthetic_dataset(lcdataset, lcset_name, obse_sampler_bdict, length_sampler_bdict, save_rootdir, **sd_kwargs)