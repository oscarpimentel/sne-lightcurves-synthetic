#!/usr/bin/env python3
# -*- coding: utf-8 -*
import sys
import argparse

sys.path.append('../fuzzy-tools')  # or just install the module
sys.path.append('../astro-lightcurves-handler')  # or just install the module
import fuzzytools as ft
from dynaconf import settings
from synthsne.synthetic_datasets import generate_synthetic_dataset
from synthsne import _C
from synthsne.distr_fittings import ObsErrorConditionalSampler
from synthsne.plots.samplers import plot_obse_samplers


# parser and settings
parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--method', type=str)
parser.add_argument('--kf', type=str)
parser.add_argument('--setn', type=str, default='train')
main_args = parser.parse_args()


# load filedir
filedir = '../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe.splcds'
filedict = ft.files.get_dict_from_filedir(filedir)
rootdir = filedict['_rootdir']
cfilename = filedict['_cfilename']
survey = filedict['survey']
lcdataset = ft.files.load_pickle(filedir)
print(lcdataset)
lcset_name = f'{main_args.kf}@{main_args.setn}'
lcset = lcdataset[lcset_name]
lcset_info = lcset.get_info()
band_names = lcset_info['band_names']
class_names = lcset_info['class_names']


# export generators
obse_sampler_bdict_full = {b: ObsErrorConditionalSampler(lcset, b) for b in band_names}
plot_obse_samplers(lcset_name, lcset_info, obse_sampler_bdict_full, original_space=1, add_samples=0, save_filedir=f'{settings.SAVE_PATH}/obse_sampler/{cfilename}/{lcset_name}/10.pdf')
plot_obse_samplers(lcset_name, lcset_info, obse_sampler_bdict_full, original_space=0, add_samples=0, save_filedir=f'{settings.SAVE_PATH}/obse_sampler/{cfilename}/{lcset_name}/00.pdf')
plot_obse_samplers(lcset_name, lcset_info, obse_sampler_bdict_full, original_space=1, add_samples=1, save_filedir=f'{settings.SAVE_PATH}/obse_sampler/{cfilename}/{lcset_name}/11.pdf')
ft.files.save_pickle(f'{settings.SAVE_PATH}/obse_sampler/{cfilename}/{lcset_name}/obse_sampler_bdict_full.d', obse_sampler_bdict_full)
obse_sampler_bdict = ft.dicts.along_dict_obj_method(obse_sampler_bdict_full, 'clean')
ft.files.save_pickle(f'{settings.SAVE_PATH}/obse_sampler/{cfilename}/{lcset_name}/obse_sampler_bdict.d', obse_sampler_bdict)


# generate synth curves
uses_estw = main_args.method.split('-')[-1] == 'estw'
ssne_save_rootdir = f'{settings.SAVE_PATH}/ssne/{main_args.method}/{cfilename}/{lcset_name}'
figs_save_rootdir = f'{settings.SAVE_PATH}/ssne_figs/{main_args.method}/{cfilename}/{lcset_name}'
generate_synthetic_dataset(lcset_name, lcset, obse_sampler_bdict, uses_estw, ssne_save_rootdir, figs_save_rootdir,
                           method=main_args.method,
                           synthetic_samples_per_curve=_C.SYNTH_SAMPLES_PER_CURVE,
                           mcmc_priors=ft.files.load_pickle(f'{settings.SAVE_PATH}/mcmc_priors/{cfilename}/{lcset_name}/mcmc_priors.d', return_none_if_missing=True),
                           )
