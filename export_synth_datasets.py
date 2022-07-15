#!/usr/bin/env python3
# -*- coding: utf-8 -*
import sys
import argparse

sys.path.append('../fuzzy-tools')  # or just install the module
sys.path.append('../astro-lightcurves-handler')  # or just install the module
import fuzzytools.files as fcfiles
from fuzzytools.prints import print_big_bar
from fuzzytools.files import load_pickle, save_pickle, get_dict_from_filedir
from fuzzytools.progress_bars import ProgressBar
from synthsne import _C


# parser and settings
parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--method', type=str)
parser.add_argument('--setn', type=str, default='train')
main_args = parser.parse_args()
print_big_bar()


filedir = '../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe.splcds'
filedict = get_dict_from_filedir(filedir)
rootdir = filedict['_rootdir']
cfilename = filedict['_cfilename']
survey = filedict['survey']
lcdataset = load_pickle(filedir)
print(lcdataset)


new_lcdataset = lcdataset.copy()  # copy with all original lcsets
for kf in lcdataset.kfolds:
    lcset_name = f'{kf}@{main_args.setn}'
    lcset = new_lcdataset[lcset_name]
    synth_rootdir = f'../save/ssne/{main_args.method}/{cfilename}/{lcset_name}'
    print(f'synth_rootdir={synth_rootdir}')
    synth_lcset = lcset.copy({})  # copy
    filedirs = fcfiles.get_filedirs(synth_rootdir, fext='ssne')
    bar = ProgressBar(len(filedirs))

    for filedir in filedirs:
        d = fcfiles.load_pickle(filedir)
        lcobj_name = d['lcobj_name']
        bar(f'lcset_name={lcset_name}; lcobj_name={lcobj_name}')

        for k, new_lcobj in enumerate(d['new_lcobjs']):
            synth_lcset.set_lcobj(f'{lcobj_name}.{k+1}', new_lcobj)

    bar.done()
    synth_lcset.reset()
    new_lcset_name = f'{lcset_name}.{main_args.method}'
    new_lcdataset.set_lcset(new_lcset_name, synth_lcset)

save_rootdir = f'{rootdir}'
save_filedir = f'{save_rootdir}/{cfilename}~method={main_args.method}.{_C.EXT_SPLIT_LIGHTCURVE}'
save_pickle(save_filedir, new_lcdataset)
