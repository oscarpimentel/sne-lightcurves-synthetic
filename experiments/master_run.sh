#!/bin/bash
SECONDS=0
clear

methods=(
	linear-fstw
	bspline-fstw
	# spm-mle-fstw
	# spm-mle-estw
	spm-mcmc-fstw
	spm-mcmc-estw
	)
for method in "${methods[@]}"; do
	script="python generate_synth_objs.py --method $method"
	script="python export_synth_datasets.py --method $method"
	echo "$script"; eval "$script"
done

mins=$((SECONDS/60))
echo echo "Time Elapsed : ${mins} minutes"