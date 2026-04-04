#!/bin/bash

# load analysis3 conda environment
module purge
module use /g/data/dk92/apps/Modules/modulefiles/
module load pet/0.4.0

# root directory for this repo
export ROOT=/home/548/${USER}/repos/solar-nowcast
export MODULES=${ROOT}/modules

# append python path
export PYTHONPATH=${MODULES}:${PYTHONPATH}