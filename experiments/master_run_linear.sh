#!/bin/bash
python generate_synth_objs.py -method linear-fstw
python generate_synth_objs.py -method bspline-fstw
python generate_synth_objs.py -method spm-mle-fstw
python generate_synth_objs.py -method spm-mle-estw