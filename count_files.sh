#!/bin/bash
now=$(date +"%T")
maxdepth=3
echo -e "\e[7m$now\e[27m"
cd save

cd ssne
ext=".ssne"
sh_path=$(realpath “${BASH_SOURCE:-$0}”)
path=$(dirname $sh_path)
echo -e "\e[7mcounting $ext files in $path:\e[27m"
find . -maxdepth $maxdepth -type d | while read -r dir
do printf "%s:\t" "$dir"; find "$dir" -type f -name "*$ext" | wc -l; done
cd ..

cd ssne_figs
ext=".pdf"
sh_path=$(realpath “${BASH_SOURCE:-$0}”)
path=$(dirname $sh_path)
echo -e "\e[7mcounting $ext files in $path:\e[27m"
find . -maxdepth $maxdepth -type d | while read -r dir
do printf "%s:\t" "$dir"; find "$dir" -type f -name "*$ext" | wc -l; done
cd ..
