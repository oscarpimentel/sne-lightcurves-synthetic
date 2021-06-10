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
from synthsne import C_
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
methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mle-estw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if main_args.method=='.' else main_args.method
methods = [methods] if isinstance(methods, str) else methods
setns = [str(setn) for setn in ['train', 'val']] if main_args.setn=='.' else main_args.setn
setns = [setns] if isinstance(setns, str) else setns

for setn in setns:
	for kf in kfs:
		for method in methods:
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
			sd_kwargs = {
				'synthetic_samples_per_curve':C_.SYNTH_SAMPLES_PER_CURVE,
				'method':method,
				'sne_specials_df':pd.read_csv(f'../data/{survey}/sne_specials.csv'),
				'mcmc_priors':load_pickle(f'../save/mcmc_priors/{cfilename}/{lcset_name}/mcmc_priors.d', return_none_if_missing=True),
			}
			uses_estw = method.split('-')[-1]=='estw'
			ssne_save_rootdir = f'../save/ssne/{method}/{cfilename}/{lcset_name}'
			figs_save_rootdir = f'../save/ssne_figs/{method}/{cfilename}/{lcset_name}'
			generate_synthetic_dataset(lcset_name, lcset, obse_sampler_bdict, uses_estw, ssne_save_rootdir, figs_save_rootdir, **sd_kwargs)

			'''
			### generate mcmc priors
			if method in ['spm-mle-fstw']:
				spm_classes = {
					'A':'GammaP',
					't0':'NormalP',
					'gamma':'GammaP',
					'f':'UniformP',
					'trise':'GammaP',
					'tfall':'GammaP',
				}
				mcmc_priors_full = nested_dict()
				for c in class_names:
					for b in band_names:
						for spm_p in spm_classes.keys():
							spm_p_samples = sms.get_spm_args(ssne_save_rootdir, spm_p, b, c)
							#print(spm_p_samples)
							mp_kwargs = {}
							if spm_p=='A':
								mp_kwargs = {'floc':0}
							mcmc_prior = getattr(mp, spm_classes[spm_p])(spm_p_samples, **mp_kwargs)
							mcmc_priors_full[b][c][spm_p] = mcmc_prior
							plot_mcmc_prior(mcmc_prior, spm_p, b, c, save_filedir=f'../save/mcmc_priors/{cfilename}/{lcset_name}/{c}_{b}_{spm_p}.png')
				
				mcmc_priors_full = mcmc_priors_full.to_dict()

				save_pickle(f'../save/mcmc_priors/{cfilename}/{lcset_name}/mcmc_priors_full.d', mcmc_priors_full)
				mcmc_priors = along_dict_obj_method(mcmc_priors_full, 'clean')
				save_pickle(f'../save/mcmc_priors/{cfilename}/{lcset_name}/mcmc_priors.d', mcmc_priors)
			'''