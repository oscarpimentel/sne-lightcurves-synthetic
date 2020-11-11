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

	root_folder = '../../astro-lightcurves-handler/save'
	filedirs = search_for_filedirs(root_folder, fext=C_.EXT_SPLIT_LIGHTCURVE)

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle
	from flamingchoripan.files import get_dict_from_filedir
	from lchandler import C_

	def load_lcdataset(filename):
		assert filename.split('.')[-1]==C_.EXT_SPLIT_LIGHTCURVE
		return load_pickle(filename)

	filedir = '../../astro-lightcurves-handler/save/alerceZTFv5.1/survey-alerceZTFv5.1_bands-gr_mode-onlySNe_kfid-0.splcds'
	filedir = '../../astro-lightcurves-handler/save/alerceZTFv7.1/survey-alerceZTFv7.1_bands-gr_mode-onlySNe_kfid-0.splcds'

	filedic = get_dict_from_filedir(filedir)
	root_folder = filedic['*rootdir*']
	cfilename = filedic['*cfilename*']
	lcdataset = load_lcdataset(filedir)
	print(lcdataset['raw'].keys())
	print(lcdataset['raw'].get_random_lcobj(False).keys())
	print(lcdataset)

	###################################################################################################################################################
	from synthsne.distr_fittings import ObsErrorConditionalSampler
	from synthsne.plots.samplers import plot_obse_samplers

	set_name = 'train'
	band_names = lcdataset[set_name].band_names
	obse_sampler_bdict = {b:ObsErrorConditionalSampler(lcdataset, set_name, b) for b in band_names}
	plot_obse_samplers(lcdataset, set_name, obse_sampler_bdict, original_space=1)
	plot_obse_samplers(lcdataset, set_name, obse_sampler_bdict, original_space=1, add_samples=1)
	plot_obse_samplers(lcdataset, set_name, obse_sampler_bdict, original_space=0)

	###################################################################################################################################################
	from synthsne.distr_fittings import CurveLengthSampler
	from synthsne.plots.samplers import plot_length_samplers

	set_name = 'train'
	band_names = lcdataset[set_name].band_names
	offset = 5
	length_sampler_bdict = {b:CurveLengthSampler(lcdataset, set_name, b, offset) for b in band_names}
	plot_length_samplers(length_sampler_bdict, lcdataset, set_name)

	###################################################################################################################################################
	from synthsne.generators.synthetic_datasets import generate_synthetic_dataset

	ignored_lcobj_names = [ # pass to .txt?
		'ZTF18aajkivu',
		'ZTF20aacbwbm',
	]
	sd_kwargs = {
		'synthetic_samples_per_curve':64,
		'method':main_args.method,
		'ignored_lcobj_names':ignored_lcobj_names,
		'save_rootdir':'../save',
	}
	errors = generate_synthetic_dataset(lcdataset, 'train', obse_sampler_bdict, length_sampler_bdict, **sd_kwargs)