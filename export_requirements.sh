#!/bin/bash
source ~/.bashrc
env_name="lchandler"
conda activate $env_name

rm requirements.txt
pip freeze > requirements.txt
python --version
rm environment.yml
conda env export > environment.yml