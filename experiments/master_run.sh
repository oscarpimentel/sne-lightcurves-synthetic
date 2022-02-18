#!/bin/bash
clear
SECONDS=0
run_script(){
	echo "$1"; eval "$1";
}

###################################################################################################################################################
methods=(
	spm-mcmc-estw
	spm-mcmc-fstw
	linear-fstw
	bspline-fstw
	)
for method in "${methods[@]}"; do
	for kf in {0..4}; do # 0..4
		run_script "python generate_synth_objs.py --method $method --kf $kf"
		:
	done
	# run_script "python export_synth_datasets.py --method $method"
	:
done

###################################################################################################################################################
mins=$((SECONDS/60))
echo echo "time elapsed=${mins} [mins]"