#!/usr/bin/env python3
# -*- coding: utf-8 -*
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../fuzzy-tools') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module

###################################################################################################################################################
import argparse
from fuzzytools.prints import print_big_bar

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--method',  type=str, default='.')
parser.add_argument('--kf',  type=str, default='.')
parser.add_argument('--setn',  type=str, default='train')
main_args = parser.parse_args()
print_big_bar()

###################################################################################################################################################
import numpy as np
from fuzzytools.files import load_pickle, save_pickle, get_dict_from_filedir

filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe.splcds'
filedict = get_dict_from_filedir(filedir)
rootdir = filedict['_rootdir']
cfilename = filedict['_cfilename']
survey = filedict['survey']
lcdataset = load_pickle(filedir)
print(lcdataset)

###################################################################################################################################################
from synthsne.synthetic_datasets import generate_synthetic_dataset
import pandas as pd
import numpy as np
from synthsne import _C
import fuzzytools.files as ff
from fuzzytools.progress_bars import ProgressBar
from fuzzytools.files import load_pickle, save_pickle
from synthsne.distr_fittings import ObsErrorConditionalSampler
from synthsne.plots.samplers import plot_obse_samplers
from synthsne.plots.mcmc import plot_mcmc_prior
from fuzzytools.dicts import along_dict_obj_method
from nested_dict import nested_dict
import synthsne.generators.mcmc_priors as mp
from synthsne.results import synth_method_statistics as sms

kfs = lcdataset.kfolds if main_args.kf=='.' else main_args.kf
kfs = [kfs] if isinstance(kfs, str) else kfs
setns = [str(setn) for setn in ['train', 'val']] if main_args.setn=='.' else main_args.setn
setns = [setns] if isinstance(setns, str) else setns

for setn in setns:
	for kf in kfs:
		lcset_name = f'{kf}@{setn}'
		lcset = lcdataset[lcset_name]
		lcset_info = lcset.get_info()
		band_names = lcset_info['band_names']
		class_names = lcset_info['class_names']

		### export generators
		obse_sampler_bdict_full = {b:ObsErrorConditionalSampler(lcset, b) for b in band_names}
		plot_obse_samplers(lcset_name, lcset_info, obse_sampler_bdict_full, original_space=1, add_samples=0, save_filedir=f'../save/obse_sampler/{cfilename}/{lcset_name}/10.png')
		plot_obse_samplers(lcset_name, lcset_info, obse_sampler_bdict_full, original_space=0, add_samples=0, save_filedir=f'../save/obse_sampler/{cfilename}/{lcset_name}/00.png')
		plot_obse_samplers(lcset_name, lcset_info, obse_sampler_bdict_full, original_space=1, add_samples=1, save_filedir=f'../save/obse_sampler/{cfilename}/{lcset_name}/11.png')

		save_pickle(f'../save/obse_sampler/{cfilename}/{lcset_name}/obse_sampler_bdict_full.d', obse_sampler_bdict_full)
		obse_sampler_bdict = along_dict_obj_method(obse_sampler_bdict_full, 'clean')
		save_pickle(f'../save/obse_sampler/{cfilename}/{lcset_name}/obse_sampler_bdict.d', obse_sampler_bdict)

		### generate synth curves
		uses_estw = main_args.method.split('-')[-1]=='estw'
		ssne_save_rootdir = f'../save/ssne/{main_args.method}/{cfilename}/{lcset_name}'
		figs_save_rootdir = f'../save/ssne_figs/{main_args.method}/{cfilename}/{lcset_name}'
		generate_synthetic_dataset(lcset_name, lcset, obse_sampler_bdict, uses_estw, ssne_save_rootdir, figs_save_rootdir,
			synthetic_samples_per_curve=_C.SYNTH_SAMPLES_PER_CURVE,
			method=main_args.method,
			sne_specials_df=pd.read_csv(f'../data/{survey}/sne_specials.csv'),
			mcmc_priors=load_pickle(f'../save/mcmc_priors/{cfilename}/{lcset_name}/mcmc_priors.d', return_none_if_missing=True),
			)