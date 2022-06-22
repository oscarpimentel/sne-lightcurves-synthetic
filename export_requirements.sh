#!/bin/bash
rm requirements.txt
pip freeze > requirements.txt
rm environment.yml
conda env export > environment.yml