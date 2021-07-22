#!/bin/bash
clear
SECONDS=0
run_script(){
	echo "$1"; eval "$1";
}

###################################################################################################################################################
methods=(
	linear-fstw
	bspline-fstw
	# spm-mle-fstw
	# spm-mle-estw
	spm-mcmc-fstw
	spm-mcmc-estw
	)
for method in "${methods[@]}"; do
	run_script "python generate_synth_objs.py --method $method"
	run_script "python export_synth_datasets.py --method $method"
done

###################################################################################################################################################
mins=$((SECONDS/60))
echo echo "Time Elapsed : ${mins} minutes"