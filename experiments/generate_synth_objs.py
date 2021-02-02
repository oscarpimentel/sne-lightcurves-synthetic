#!/usr/bin/env python3
# -*- coding: utf-8 -*
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
	from flamingchoripan.files import search_for_filedirs
	from synthsne import C_

	root_folder = '../../surveys-save'
	filedirs = search_for_filedirs(root_folder, fext=C_.EXT_SPLIT_LIGHTCURVE)

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle, get_dict_from_filedir

	filedir = f'../../surveys-save/alerceZTFv7.1/survey=alerceZTFv7.1°bands=gr°mode=onlySNe.splcds'
	filedict = get_dict_from_filedir(filedir)
	root_folder = filedict['*rootdir*']
	cfilename = filedict['*cfilename*']
	survey = filedict['survey']
	lcdataset = load_pickle(filedir)
	print(lcdataset)

	###################################################################################################################################################
	from synthsne.generators.synthetic_datasets import generate_synthetic_dataset
	import pandas as pd
	from synthsne import C_
	import flamingchoripan.files as ff
	from flamingchoripan.progress_bars import ProgressBar
	from flamingchoripan.files import load_pickle, save_pickle

	kfs = [str(kf) for kf in range(0,3)] if main_args.kf=='.' else main_args.kf
	kfs = [kfs] if isinstance(kfs, str) else kfs
	methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mle-estw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if main_args.method=='.' else main_args.method
	methods = [methods] if isinstance(methods, str) else methods
	setns = [str(setn) for setn in ['train', 'val']] if main_args.setn=='.' else main_args.setn
	setns = [setns] if isinstance(setns, str) else setns

	for setn in setns:
		for kf in kfs:
			for method in methods:
				lcset_name = f'{kf}@{setn}'
				save_rootdir = f'../save/{survey}/{cfilename}/{lcset_name}'
				sd_kwargs = {
					'synthetic_samples_per_curve':C_.SYNTH_SAMPLES_PER_CURVE,
					'method':method,
					'sne_specials_df':pd.read_csv(f'../data/{survey}/sne_specials.csv'),
				}
				samplers = load_pickle(f'{save_rootdir}/samplers.{C_.EXT_SAMPLER}')
				obse_sampler_bdict = samplers['obse_sampler_bdict']
				length_sampler_bdict = samplers['length_sampler_bdict']
				uses_estw = method.split('-')[-1]=='estw'
				generate_synthetic_dataset(lcdataset, lcset_name, obse_sampler_bdict, length_sampler_bdict, uses_estw, save_rootdir, **sd_kwargs)