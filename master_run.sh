#!/bin/bash
reset
SECONDS=0
run_python_script(){
    now=$(date +"%T")
    echo -e "\e[7mrunning script ($now)\e[27m python $1"
    # eval "python $1"  # to perform serial runs
    eval "python $1 > /dev/null 2>&1" &  # to perform parallel runs
}
intexit(){
    kill -HUP -$$
}
hupexit(){
    echo
    echo "Interrupted"
    exit
}
trap hupexit HUP
trap intexit INT
echo -e "\e[7mrunning master_run... (ctrl+c to interrupt)\e[27m $1"
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
methods=(
    linear-fstw
    spm-mcmc-estw
    spm-mcmc-fstw
    bspline-fstw
)
kfs=({0..4})  # k-folds
for method in "${methods[@]}"; do
    for kf in "${kfs[@]}"; do
        run_python_script "generate_synth_objs.py --method $method --kf $kf"
        :
    done
	wait
    # run_python_script "export_synth_datasets.py --method $method"  # export all kf results
    :
    wait
done
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
mins=$((SECONDS/60))
echo -e "\e[7mtime elapsed=${mins}[mins]\e[27m"