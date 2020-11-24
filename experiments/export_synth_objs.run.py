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
	parser.add_argument('-method',  type=str, default=None, help='method')
	main_args = parser.parse_args()
	print_big_bar()

	###################################################################################################################################################
	from flamingchoripan.files import search_for_filedirs
	from lchandler import C_

	root_folder = '../../surveys-save'
	filedirs = search_for_filedirs(root_folder, fext=C_.EXT_SPLIT_LIGHTCURVE)

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle
	from flamingchoripan.files import get_dict_from_filedir
	from lchandler import C_

	def load_lcdataset(filename):
		assert filename.split('.')[-1]==C_.EXT_SPLIT_LIGHTCURVE
		return load_pickle(filename)

	filedir = '../../surveys-save/alerceZTFv7.1/survey-alerceZTFv7.1_bands-gr_mode-onlySNe_kfid-0.splcds'

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

	save_rootdir = f'../save/{survey}/{cfilename}'
	sd_kwargs = {
		'synthetic_samples_per_curve':64,
		'method':main_args.method,
		'sne_specials_df':pd.read_csv(f'../data/{survey}/sne_specials.csv'),
	}
	samplers = load_pickle(f'{save_rootdir}/samplers.s')
	obse_sampler_bdict = samplers['obse_sampler_bdict']
	length_sampler_bdict = samplers['length_sampler_bdict']
	errors = generate_synthetic_dataset(lcdataset, 'train', obse_sampler_bdict, length_sampler_bdict, save_rootdir, **sd_kwargs)